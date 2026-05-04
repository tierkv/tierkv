"""vLLM connector configuration — reads from tierkv.toml or inline dict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class VllmConnectorConfig:
    # Vault nodes
    kv_cold_host: str = "127.0.0.1"
    kv_cold_port: int = 50051
    ssm_cold_host: Optional[str] = None
    ssm_cold_port: int = 50052

    # vLLM alignment
    block_size: int = 16

    # TurboQuant
    turbo_quant: bool = True
    kv_dim: int = 128  # head_dim: Llama3=128, Qwen3=256, Mistral=128

    # Layer type mapping for hybrid MoE models
    # group_id (int) -> layer_type (str)
    layer_type_map: dict = field(default_factory=dict)

    # Concurrency
    max_inflight_stores: int = 8
    max_inflight_promotes: int = 4

    @classmethod
    def from_toml(cls, path: str) -> VllmConnectorConfig:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        vault = raw.get("vault", {})
        cluster = raw.get("cluster", {})
        inference = raw.get("inference", {})
        tierkv = raw.get("tierkv", {})

        # Support both flat [vault] keys and nested [cluster.kv_cold]
        kv_cold = cluster.get("kv_cold", {})
        ssm_cold = cluster.get("ssm_cold", {})

        return cls(
            kv_cold_host=vault.get("kv_cold_host", kv_cold.get("host", cls.kv_cold_host)),
            kv_cold_port=int(vault.get("kv_cold_port", kv_cold.get("port", cls.kv_cold_port))),
            ssm_cold_host=vault.get("ssm_cold_host", ssm_cold.get("host")),
            ssm_cold_port=int(vault.get("ssm_cold_port", ssm_cold.get("port", cls.ssm_cold_port))),
            block_size=int(tierkv.get("block_size", cls.block_size)),
            turbo_quant=bool(tierkv.get("turbo_quant", cls.turbo_quant)),
            kv_dim=int(tierkv.get("kv_dim", inference.get("kv_dim", cls.kv_dim))),
            layer_type_map=tierkv.get("layer_type_map", {}),
            max_inflight_stores=int(tierkv.get("max_inflight_stores", cls.max_inflight_stores)),
            max_inflight_promotes=int(tierkv.get("max_inflight_promotes", cls.max_inflight_promotes)),
        )

    @classmethod
    def from_dict(cls, d: dict) -> VllmConnectorConfig:
        """For --kv-connector-extra-config JSON path."""
        config_path = d.get("config_path")
        if config_path:
            return cls.from_toml(config_path)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
