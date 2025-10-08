"""Weights & Biases (W&B) utility helpers for TWIST training."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:  # Optional dependency – handled gracefully when missing
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - dependency may be absent in some setups
    wandb = None  # type: ignore

try:  # Optional dependency for YAML based configuration
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - dependency may be absent in some setups
    yaml = None  # type: ignore


_REPO_ROOT = Path(__file__).resolve().parents[3]
_WANDB_KEY_PATH = _REPO_ROOT / "wandb_key"
_WANDB_CONFIG_PATH = _REPO_ROOT / "config" / "wandb_config.yaml"

_WANDB_RUN = None
_WANDB_DISABLED_REASON = "not initialized"
_WANDB_SAVE_CONFIG = True


def _resolve_config_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (_WANDB_CONFIG_PATH.parent / path).resolve()
    return path


@lru_cache(maxsize=1)
def _load_wandb_config() -> dict[str, Any]:
    base_cfg: dict[str, Any] = {
        "wandb": {
            "api_key": None,
            "entity": None,
            "project": None,
            "mode": "online",
            "dir": None,
            "tags": [],
            "settings": {
                "save_config_files": True,
            },
        },
        "fallback": {},
    }

    if not _WANDB_CONFIG_PATH.exists():
        return base_cfg

    if yaml is None:
        print("⚠️  PyYAML not installed – using default W&B config")
        return base_cfg

    try:
        loaded = yaml.safe_load(_WANDB_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️  Failed to read W&B config: {exc}")
        return base_cfg

    if not isinstance(loaded, dict):
        return base_cfg

    wandb_cfg = loaded.get("wandb")
    if isinstance(wandb_cfg, dict):
        merged_settings = dict(base_cfg["wandb"].get("settings", {}))
        user_settings = wandb_cfg.get("settings")
        if isinstance(user_settings, dict):
            merged_settings.update(user_settings)

        merged_cfg = dict(base_cfg["wandb"])
        merged_cfg.update({k: v for k, v in wandb_cfg.items() if k != "settings"})
        merged_cfg["settings"] = merged_settings
        base_cfg["wandb"] = merged_cfg

    fallback_cfg = loaded.get("fallback")
    if isinstance(fallback_cfg, dict):
        base_cfg["fallback"] = fallback_cfg

    return base_cfg


def read_wandb_key(key_path: Optional[str] = None) -> str:
    """Return the W&B API key from file, or an empty string when missing."""

    path = Path(key_path) if key_path else _WANDB_KEY_PATH
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"🔕 W&B key file not found at: {path}")
        return ""
    except OSError as exc:  # pragma: no cover - defensive
        print(f"⚠️  Unable to read W&B key: {exc}")
        return ""

    if not key:
        print(f"⚠️  W&B key file at {path} is empty")
    return key


def setup_wandb(project_name: str,
                experiment_id: str,
                robot_type: str = "unknown",
                force_disabled: bool = False,
                debug: bool = False) -> bool:
    """Initialise a W&B run based on repository configuration."""

    global _WANDB_RUN, _WANDB_DISABLED_REASON, _WANDB_SAVE_CONFIG

    if _WANDB_RUN is not None:
        return True

    if force_disabled:
        _WANDB_DISABLED_REASON = "manual override"
        print("🔕 W&B disabled via --no-wandb flag")
        return False

    if debug:
        _WANDB_DISABLED_REASON = "debug mode"
        print("🔕 W&B disabled in debug mode")
        return False

    if wandb is None:
        _WANDB_DISABLED_REASON = "wandb package not installed"
        print("⚠️  W&B Python package not available – skipping instrumentation")
        return False

    config = _load_wandb_config()
    wandb_cfg: dict[str, Any] = dict(config.get("wandb", {}))

    mode = str(wandb_cfg.get("mode", "online")).lower().strip()
    if mode == "disabled":
        _WANDB_DISABLED_REASON = "configuration disables W&B"
        print("🔕 W&B disabled via config/wandb_config.yaml")
        return False

    key = str(wandb_cfg.get("api_key") or "").strip() or os.environ.get("WANDB_API_KEY") or read_wandb_key()
    if key:
        os.environ.setdefault("WANDB_API_KEY", key)
    elif mode == "online":
        _WANDB_DISABLED_REASON = "missing API key"
        print("� W&B disabled – no API key configured")
        return False

    project = str(wandb_cfg.get("project") or project_name or "").strip()
    entity = str(wandb_cfg.get("entity") or "").strip() or None

    dir_value = wandb_cfg.get("dir")
    run_dir = None
    if isinstance(dir_value, str) and dir_value.strip():
        run_dir = _resolve_config_path(dir_value.strip())
        run_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(run_dir))

    tags: set[str] = set()
    if robot_type and robot_type != "unknown":
        tags.add(robot_type)
    user_tags = wandb_cfg.get("tags")
    if isinstance(user_tags, Sequence):
        tags.update(str(tag) for tag in user_tags if str(tag).strip())

    init_kwargs: dict[str, Any] = {
        "project": project or None,
        "entity": entity,
        "name": experiment_id,
        "config": {
            "robot_type": robot_type,
            "experiment_id": experiment_id,
        },
    }

    if tags:
        init_kwargs["tags"] = sorted(tags)

    if run_dir is not None:
        init_kwargs["dir"] = str(run_dir)

    if mode == "offline":
        init_kwargs["mode"] = "offline"

    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    try:
        _WANDB_RUN = wandb.init(**init_kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        _WANDB_DISABLED_REASON = f"wandb.init failed: {exc}"
        print(f"⚠️  Failed to initialise W&B: {exc}")
        _WANDB_RUN = None
        return False

    if _WANDB_RUN is None:
        _WANDB_DISABLED_REASON = "wandb returned None"
        print("🔕 W&B run not created (check mode setting)")
        return False

    settings = wandb_cfg.get("settings")
    _WANDB_SAVE_CONFIG = bool(settings.get("save_config_files", True)) if isinstance(settings, Mapping) else True

    _WANDB_DISABLED_REASON = ""
    print(f"🚀 W&B run initialised: {project}/{experiment_id}")
    return True


def _collect_config_candidates(robot_type: str, envs_dir: str | Path) -> Iterable[Path]:
    env_root = Path(envs_dir)
    if env_root.is_dir():
        robot_cfg_dir = env_root / robot_type
        if robot_cfg_dir.is_dir():
            for pattern in ("*config*.py", "*.yaml", "*.yml"):
                yield from robot_cfg_dir.glob(pattern)

    if _WANDB_CONFIG_PATH.exists():
        yield _WANDB_CONFIG_PATH


def save_config_files(robot_type: str, envs_dir: str) -> None:
    """Upload relevant configuration files to the active W&B run."""

    if wandb is None or _WANDB_RUN is None:
        return

    if not _WANDB_SAVE_CONFIG:
        return

    artifact = wandb.Artifact(name=f"{robot_type}_configs", type="config")
    files_added = 0

    for candidate in _collect_config_candidates(robot_type, envs_dir):
        if candidate.is_file():
            artifact.add_file(str(candidate))
            files_added += 1

    if files_added == 0:
        return

    try:
        _WANDB_RUN.log_artifact(artifact)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️  Failed to log config artifact: {exc}")


def log_metrics(metrics: Mapping[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
    """Proxy to ``wandb.log`` when a run is active."""

    if wandb is None or _WANDB_RUN is None:
        return

    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping of names to values")

    _WANDB_RUN.log(dict(metrics), step=step, commit=commit)


def finish_wandb() -> None:
    """Gracefully close the active W&B run."""

    global _WANDB_RUN

    if wandb is None or _WANDB_RUN is None:
        return

    try:
        _WANDB_RUN.finish()
        print("🏁 W&B run closed")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️  Failed to close W&B run: {exc}")
    finally:
        _WANDB_RUN = None


def get_wandb_status() -> str:
    """Return human-readable status information about the current W&B run."""

    if _WANDB_RUN is not None:
        return "running"

    return f"disabled: {_WANDB_DISABLED_REASON}" if _WANDB_DISABLED_REASON else "disabled"