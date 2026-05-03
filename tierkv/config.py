"""
tierkv configuration loader.

Searches for tierkv.toml in:
  1. Path from TIERKV_CONFIG environment variable
  2. Current working directory (./tierkv.toml)
  3. ~/.config/tierkv/tierkv.toml

See tierkv.toml.example for all available options.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # stdlib Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli for Python 3.9/3.10
    except ImportError:
        tomllib = None  # type: ignore


@dataclass
class NodeConfig:
    host: str
    port: int

    @property
    def addr(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class ClusterConfig:
    role: str = "inference"
    kv_cold: NodeConfig = field(default_factory=lambda: NodeConfig("127.0.0.1", 50051))
    ssm_cold: NodeConfig = field(default_factory=lambda: NodeConfig("127.0.0.1", 50051))
    recompute: NodeConfig = field(default_factory=lambda: NodeConfig("127.0.0.1", 50052))


@dataclass
class InferenceConfig:
    exo_path: str = ""
    log_file: str = "/tmp/tierkv.log"
    memory_threshold: float = 0.60
    kv_dim: int = 256


@dataclass
class VaultConfig:
    port: int = 50051
    persist_path: str = ""


@dataclass
class TierkvConfig:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    vault: VaultConfig = field(default_factory=VaultConfig)


def _default_config() -> TierkvConfig:
    return TierkvConfig()


def _parse(data: dict) -> TierkvConfig:
    cfg = TierkvConfig()

    cluster_data = data.get("cluster", {})
    cfg.cluster.role = cluster_data.get("role", cfg.cluster.role)

    for tier_name in ("kv_cold", "ssm_cold", "recompute"):
        tier_data = cluster_data.get(tier_name, {})
        node = getattr(cfg.cluster, tier_name)
        node.host = tier_data.get("host", node.host)
        node.port = int(tier_data.get("port", node.port))

    inf = data.get("inference", {})
    cfg.inference.exo_path = inf.get("exo_path", cfg.inference.exo_path)
    cfg.inference.log_file = inf.get("log_file", cfg.inference.log_file)
    cfg.inference.memory_threshold = float(inf.get("memory_threshold", cfg.inference.memory_threshold))
    cfg.inference.kv_dim = int(inf.get("kv_dim", cfg.inference.kv_dim))

    vault = data.get("vault", {})
    cfg.vault.port = int(vault.get("port", cfg.vault.port))
    cfg.vault.persist_path = vault.get("persist_path", cfg.vault.persist_path)

    return cfg


def load_config(path: str | None = None) -> TierkvConfig:
    """
    Load tierkv configuration from a TOML file.

    Search order:
      1. Explicit path argument
      2. TIERKV_CONFIG environment variable
      3. ./tierkv.toml (current working directory)
      4. ~/.config/tierkv/tierkv.toml

    Returns a TierkvConfig with defaults if no config file is found.
    """
    candidates = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get("TIERKV_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "tierkv.toml")
    candidates.append(Path.home() / ".config" / "tierkv" / "tierkv.toml")

    for candidate in candidates:
        if candidate.is_file():
            if tomllib is None:
                raise ImportError(
                    "No TOML library available. "
                    "Install tomli: pip install tomli  (required for Python < 3.11)"
                )
            with open(candidate, "rb") as f:
                data = tomllib.load(f)
            return _parse(data)

    return _default_config()
