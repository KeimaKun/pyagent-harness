"""
harness/config_loader.py
========================
Loads config.yaml and provides a PII sanitizer that scrubs API keys,
local IPv4/v6 addresses, and internal hostnames from any outgoing network payload.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# ---------------------------------------------------------------------------
# PII / Secret redaction patterns
# ---------------------------------------------------------------------------
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # OpenAI-style keys: sk-xxxx
    ("API_KEY_SK", re.compile(r'\bsk-[A-Za-z0-9_\-]{20,}\b')),
    # Google API keys: AIzaSy...
    ("API_KEY_GCP", re.compile(r'\bAIza[A-Za-z0-9_\-]{35}\b')),
    # Bearer tokens
    ("BEARER_TOKEN", re.compile(r'(?i)\bBearer\s+[A-Za-z0-9_\-\.=]+\b')),
    # Generic hex secrets (32+ chars)
    ("SECRET_HEX", re.compile(r'\b[0-9a-fA-F]{32,}\b')),
    # IPv4
    ("IPV4_ADDR", re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    )),
    # IPv6 (simplified, catches common forms)
    ("IPV6_ADDR", re.compile(
        r'(?i)\b(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}\b|::(?:[0-9a-f]{1,4}:?)*\b'
    )),
    # Internal hostnames: *.local, *.internal, *.corp, *.lan, *.home
    ("INTERNAL_HOST", re.compile(
        r'\b[\w\-]+\.(?:local|internal|corp|lan|home|intranet)\b',
        re.IGNORECASE,
    )),
]


class PIISanitizer:
    """Regex-based sanitizer that redacts sensitive data from strings."""

    def __init__(self, patterns: list[tuple[str, re.Pattern[str]]] | None = None) -> None:
        self._patterns = patterns or _PATTERNS

    def sanitize(self, text: str) -> str:
        """Return *text* with all sensitive tokens replaced by redaction markers."""
        for label, pattern in self._patterns:
            text = pattern.sub(f"[REDACTED:{label}]", text)
        return text

    def __call__(self, text: str) -> str:
        return self.sanitize(text)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
class ConfigLoader:
    """Load and expose config.yaml as a typed dict-like object."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        self._path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._data: dict[str, Any] = {}
        self._sanitizer = PIISanitizer()
        self.reload()

    # ------------------------------------------------------------------
    def reload(self) -> None:
        """Re-read the YAML file from disk."""
        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")
        with self._path.open("r", encoding="utf-8") as fh:
            self._data = yaml.safe_load(fh) or {}

    # ------------------------------------------------------------------
    # Accessor helpers
    # ------------------------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        """Navigate nested keys with dot-path support, e.g. get('model', 'n_ctx')."""
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
            if node is default:
                return default
        return node

    @property
    def cloud_router_mode(self) -> str:
        return self.get("cloud_router", "mode", default="disabled")

    @property
    def model_path(self) -> str:
        return self.get("model", "path", default="")

    @property
    def model_n_gpu_layers(self) -> int:
        return int(self.get("model", "n_gpu_layers", default=-1))

    @property
    def model_n_ctx(self) -> int:
        return int(self.get("model", "n_ctx", default=8192))

    @property
    def model_temperature(self) -> float:
        return float(self.get("model", "temperature", default=0.2))

    @property
    def model_max_tokens(self) -> int:
        return int(self.get("model", "max_tokens", default=2048))

    @property
    def model_verbose(self) -> bool:
        return bool(self.get("model", "verbose", default=True))

    @property
    def model_enable_thinking(self) -> bool:
        """When False (default), strip <think>...</think> blocks from model output.
        Set to true in config.yaml only if you want the chain-of-thought preserved.
        """
        return bool(self.get("model", "enable_thinking", default=False))

    @property
    def execution_timeout(self) -> int:
        return int(self.get("execution", "timeout_seconds", default=45))

    @property
    def ast_safety_check(self) -> bool:
        return bool(self.get("execution", "ast_safety_check", default=True))

    @property
    def max_react_iterations(self) -> int:
        return int(self.get("execution", "max_react_iterations", default=10))

    @property
    def sanitizer(self) -> PIISanitizer:
        return self._sanitizer

    def sanitize_payload(self, text: str) -> str:
        """Convenience wrapper — sanitize before any outgoing network payload."""
        return self._sanitizer.sanitize(text)

    def raw(self) -> dict[str, Any]:
        return dict(self._data)


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------
_instance: ConfigLoader | None = None


def get_config(config_path: Path | str | None = None) -> ConfigLoader:
    """Return a module-level singleton ConfigLoader (lazy-initialized)."""
    global _instance
    if _instance is None:
        _instance = ConfigLoader(config_path)
    return _instance


def reset_config() -> None:
    """Reset the singleton (used in tests)."""
    global _instance
    _instance = None
