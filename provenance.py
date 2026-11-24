"""Provenance helpers: which code and which exact files produced a result."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parent


def code_version() -> dict[str, Any]:
    """Git commit of this CLI, and whether it had uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": has_uncommitted_code(git("status", "--porcelain", "--untracked-files=all")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown (not a git checkout)"}


def has_uncommitted_code(porcelain_status: str) -> bool:
    """Changed tracked files, or new Python files not yet committed (which a run may import)."""
    for line in porcelain_status.splitlines():
        if not line.startswith("??") or line.endswith(".py"):
            return True
    return False


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
