"""Project configuration: YAML settings, repo-relative paths and secrets from .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("DRAFTAI_CONFIG", ROOT / "config" / "config.yaml"))


@lru_cache(maxsize=1)
def cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Resolve a `paths.<key>` entry relative to the repo root and make sure its parent exists."""
    p = ROOT / cfg()["paths"][key]
    (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def secret(name: str) -> str:
    """Read a secret from the environment (.env). Never log the returned value."""
    load_dotenv(ROOT / ".env")
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def position_group(pos: str | None) -> str | None:
    for group, members in cfg()["positions"]["groups"].items():
        if pos in members:
            return group
    return None
