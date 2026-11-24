"""Hash-verified data access. Any tampering -> IntegrityError, pipeline rukta hai."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MANIFEST = pathlib.Path(os.getenv("DATA_MANIFEST", "config/manifest.json"))
CHUNK = 1 << 20
MANIFEST_VERSION = 2


class IntegrityError(RuntimeError):
    """Data file missing from manifest, or hash mismatch."""


# ---------------------------------------------------------------- hashing

def _normalize_key(path: str | pathlib.Path) -> str:
    """Normalize path separators so Windows and Linux CI share identical keys."""
    return pathlib.Path(path).as_posix()


def _digest(path: pathlib.Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path: pathlib.Path) -> str:
    return _digest(path, "sha256")


def md5(path: pathlib.Path) -> str:
    return _digest(path, "md5")


# --------------------------------------------------------------- manifest

def _empty() -> dict:
    return {"version": MANIFEST_VERSION, "generated_at": None, "files": {}}


def _read_manifest() -> dict:
    if not MANIFEST.exists():
        return _empty()
    content = MANIFEST.read_text(encoding="utf-8").strip()
    if not content:
        return _empty()
    raw = json.loads(content)
    # migrate flat v1 {path: sha} -> v2 {files: {path: {...}}}
    if "files" not in raw:
        raw = {
            "version": MANIFEST_VERSION,
            "generated_at": None,
            "files": {_normalize_key(k): {"sha256": v} for k, v in raw.items()},
        }
    return raw


def _write_manifest(data: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    data["version"] = MANIFEST_VERSION
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    fd, tmp = tempfile.mkstemp(dir=MANIFEST.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, MANIFEST)  # atomic, no half-written manifest
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def record(path: str | pathlib.Path, **provenance) -> str:
    """Hash a file and merge its entry into the manifest. Returns the hash."""
    p = pathlib.Path(path)
    if not p.exists():
        raise IntegrityError(f"{p} disk par nahi hai — record nahi kar sakte")
    digest = sha256(p)
    data = _read_manifest()
    data["files"][_normalize_key(p)] = {
        "sha256": digest,
        "size": p.stat().st_size,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance or {},
    }
    _write_manifest(data)
    return digest


def write_manifest(paths: list[str | pathlib.Path]) -> None:
    """Record many paths at once, preserving existing entries for others."""
    data = _read_manifest()
    for path in paths:
        p = pathlib.Path(path)
        if not p.exists():
            raise IntegrityError(f"{p} disk par nahi hai — manifest update fail")
        data["files"][_normalize_key(p)] = {
            "sha256": sha256(p),
            "size": p.stat().st_size,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    _write_manifest(data)


# --------------------------------------------------------------- checking

def expected_hash(path: str | pathlib.Path) -> str:
    key = _normalize_key(path)
    entry = _read_manifest()["files"].get(key)
    if entry is None:
        raise IntegrityError(f"{key} manifest me nahi hai — untracked data")
    return entry["sha256"]


def verify(path: str | pathlib.Path) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        raise IntegrityError(f"{p} disk par nahi hai")
    expected = expected_hash(p)
    actual = sha256(p)
    if actual != expected:
        raise IntegrityError(f"hash mismatch {p}: {expected[:12]} != {actual[:12]}")
    return actual


def verify_all() -> None:
    """CI gate: every manifest entry must exist and match."""
    manifest_data = _read_manifest()
    files = manifest_data.get("files", {})
    if not files:
        raise IntegrityError("manifest khali hai — koi bhi file tracked nahi hai")
    problems = []
    for name, entry in files.items():
        p = pathlib.Path(name)
        if not p.exists():
            problems.append(f"missing: {name}")
        elif sha256(p) != entry["sha256"]:
            problems.append(f"tampered: {name}")
    if problems:
        raise IntegrityError("integrity check failed:\n  " + "\n  ".join(problems))


def load_verified(path: str | pathlib.Path, **read_kwargs) -> pd.DataFrame:
    verify(path)
    return pd.read_csv(path, **read_kwargs)


if __name__ == "__main__":
    import sys

    if "--verify-all" in sys.argv:
        verify_all()
        print("manifest OK")
