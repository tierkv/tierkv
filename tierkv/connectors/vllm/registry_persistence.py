"""
PersistentBlockRegistry — SQLite-backed BlockRegistry.

Extends BlockRegistry with write-behind persistence so the block_hash → vault_key
mapping survives vLLM restarts. Without persistence, every vault block becomes
orphaned on restart: the vault (Rust gRPC server) retains the payloads but the
Python registry loses the mapping, making restore impossible.

Architecture
────────────
- In-memory dict remains the source of truth for all reads (zero latency).
- mark_stored / remove enqueue writes to a background thread that commits to
  SQLite in batches. This means zero write latency on the scheduling hot path,
  with blocks durable within ~100ms.
- Only "stored" records are persisted. "Pending" records are in-flight; if vLLM
  crashes mid-store the vault may or may not have received them, so they are
  always treated as cold misses after restart.
- The worker process (vLLM v1: scheduler and worker are separate OS processes)
  must open the db read-only to avoid WAL conflicts. Pass readonly=True in that
  case.

Usage
─────
    registry = PersistentBlockRegistry("/var/lib/tierkv/registry.db")
    # or for the worker process:
    registry = PersistentBlockRegistry("/var/lib/tierkv/registry.db", readonly=True)
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import sqlite3
import threading
import time
from typing import Optional

from tierkv.connectors.vllm.block_registry import BlockRecord, BlockRegistry

_log = logging.getLogger("tierkv.registry_persistence")

# Schema version — bump when columns change so we can migrate cleanly.
_SCHEMA_VERSION = 1

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS blocks (
    block_hash   BLOB PRIMARY KEY,
    vault_key    TEXT NOT NULL,
    vault_node   TEXT NOT NULL,
    layer_type   TEXT NOT NULL DEFAULT 'full_attention',
    num_tokens   INTEGER NOT NULL DEFAULT 0,
    size_bytes   INTEGER NOT NULL DEFAULT 0,
    evicted_at   REAL NOT NULL,
    tensor_hash  BLOB NOT NULL DEFAULT X'',
    context_id   TEXT,
    position     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_vault_key ON blocks(vault_key);
"""

_UPSERT = """
INSERT INTO blocks
    (block_hash, vault_key, vault_node, layer_type, num_tokens, size_bytes,
     evicted_at, tensor_hash, context_id, position)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(block_hash) DO UPDATE SET
    vault_key   = excluded.vault_key,
    vault_node  = excluded.vault_node,
    layer_type  = excluded.layer_type,
    num_tokens  = excluded.num_tokens,
    size_bytes  = excluded.size_bytes,
    evicted_at  = excluded.evicted_at,
    tensor_hash = excluded.tensor_hash,
    context_id  = excluded.context_id,
    position    = excluded.position
"""

_DELETE = "DELETE FROM blocks WHERE block_hash = ?"

# Background thread drains the queue in batches of this size.
_BATCH_SIZE = 64

# Sentinel to stop the background thread cleanly.
_STOP = object()


class PersistentBlockRegistry(BlockRegistry):
    """
    BlockRegistry with SQLite write-behind persistence.

    Args:
        db_path:  Path to the SQLite database file. Created if absent.
        readonly: Open the db read-only (use for the vLLM worker process).
        max_age_seconds: Discard records older than this on startup.
    """

    def __init__(
        self,
        db_path: str,
        readonly: bool = False,
        max_age_seconds: float = 86400.0,  # 24 hours
    ) -> None:
        super().__init__()
        self._db_path = db_path
        self._readonly = readonly

        if not readonly:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            self._init_db()
            self._queue: queue.Queue = queue.Queue()
            self._writer = threading.Thread(
                target=self._write_loop,
                name="tierkv-db-writer",
                daemon=True,
            )
            self._writer.start()
            atexit.register(self.close)

        self._load_from_db(max_age_seconds)
        _log.info(
            "[tierkv] PersistentBlockRegistry: loaded %d stored blocks from %s",
            len(self._records), db_path,
        )

    # ── Init ──────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        if self._readonly:
            uri = f"file:{self._db_path}?mode=ro"
            return sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_CREATE_TABLE)
            conn.commit()
        finally:
            conn.close()

    def _load_from_db(self, max_age_seconds: float) -> None:
        """Populate in-memory registry from disk at startup."""
        if not os.path.exists(self._db_path):
            return
        cutoff = time.time() - max_age_seconds
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT block_hash, vault_key, vault_node, layer_type, "
                    "num_tokens, size_bytes, evicted_at, tensor_hash, "
                    "context_id, position FROM blocks WHERE evicted_at > ?",
                    (cutoff,),
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError:
            # DB doesn't exist yet or no table — fresh start
            return

        with self._lock:
            for row in rows:
                (block_hash, vault_key, vault_node, layer_type, num_tokens,
                 size_bytes, evicted_at, tensor_hash, context_id, position) = row
                rec = BlockRecord(
                    block_hash=bytes(block_hash),
                    vault_key=vault_key,
                    vault_node=vault_node,
                    layer_type=layer_type,
                    num_tokens=num_tokens,
                    size_bytes=size_bytes,
                    evicted_at=evicted_at,
                    tensor_hash=bytes(tensor_hash) if tensor_hash else b"",
                    status="stored",
                    context_id=context_id,
                    position=position,
                )
                self._records[rec.block_hash] = rec
                if vault_key:
                    self._vault_index[vault_key] = rec.block_hash

    # ── Overrides ─────────────────────────────────────────────────────────────

    def mark_stored(self, block_hash: bytes) -> None:
        """Mark block as stored in-memory and enqueue a disk write."""
        super().mark_stored(block_hash)
        if self._readonly:
            return
        with self._lock:
            rec = self._records.get(block_hash)
        if rec is not None:
            self._queue.put(("upsert", rec))

    def remove(self, block_hash: bytes) -> None:
        """Remove from in-memory registry and enqueue a disk delete."""
        super().remove(block_hash)
        if not self._readonly:
            self._queue.put(("delete", block_hash))

    # ── Background writer ─────────────────────────────────────────────────────

    def _write_loop(self) -> None:
        conn = self._connect()
        try:
            while True:
                # Wait up to 2 seconds for an item so we commit promptly
                # even if close() is never called (e.g. os._exit).
                try:
                    item = self._queue.get(timeout=2.0)
                except queue.Empty:
                    # Nothing pending — loop back and wait again.
                    continue

                if item is _STOP:
                    break

                batch = [item]
                # Drain the rest of the queue without blocking
                while len(batch) < _BATCH_SIZE:
                    try:
                        nxt = self._queue.get_nowait()
                        if nxt is _STOP:
                            batch.append(nxt)
                            break
                        batch.append(nxt)
                    except queue.Empty:
                        break

                upserts: list[tuple] = []
                deletes: list[tuple] = []
                stop = False

                for item in batch:
                    if item is _STOP:
                        stop = True
                        break
                    op, payload = item
                    if op == "upsert":
                        rec: BlockRecord = payload
                        upserts.append((
                            rec.block_hash, rec.vault_key, rec.vault_node,
                            rec.layer_type, rec.num_tokens, rec.size_bytes,
                            rec.evicted_at,
                            rec.tensor_hash if rec.tensor_hash else b"",
                            rec.context_id, rec.position,
                        ))
                    elif op == "delete":
                        deletes.append((payload,))

                try:
                    if upserts:
                        conn.executemany(_UPSERT, upserts)
                    if deletes:
                        conn.executemany(_DELETE, deletes)
                    if upserts or deletes:
                        conn.commit()
                        _log.info(
                            "[tierkv] registry: committed %d upserts %d deletes to SQLite",
                            len(upserts), len(deletes),
                        )
                except sqlite3.Error as exc:
                    _log.error("[tierkv] db write error: %s", exc)
                    try:
                        conn.rollback()
                    except Exception:
                        pass

                if stop:
                    break
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def close(self) -> None:
        """Flush pending writes and stop the background thread."""
        if not self._readonly:
            self._queue.put(_STOP)
            self._writer.join(timeout=5.0)
