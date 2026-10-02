"""Provenance stamp written next to every fitted result.

The paper's cached fits could not be traced to the code that produced them: outputs are
gitignored and were overwritten in place by later runs. Every run summary now records the
git commit, whether the tree was dirty (and a hash of the diff), and the sha256 of the
input catalogue, so a number can be matched to its code and data.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_stamp(input_paths: list[Path] | None = None) -> dict[str, object]:
    diff = _git("diff", "HEAD", "--", "src", "scripts")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_src_or_scripts": bool(diff),
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else "",
        "python": sys.version.split()[0],
        "command": " ".join(sys.argv),
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            str(Path(path)): sha256_of_file(Path(path)) for path in (input_paths or []) if Path(path).exists()
        },
    }
