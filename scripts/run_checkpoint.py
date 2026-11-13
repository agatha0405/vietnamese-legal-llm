"""Publish and verify local or Drive-backed single-process run artifacts."""

from contextlib import contextmanager
from hashlib import file_digest
import json
import os
import socket
from uuid import uuid4


def sha256(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    staged = path.with_name(f".{path.name}-{uuid4().hex}")
    with staged.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    staged.rename(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def publish(staged, target, identity, step, required):
    if target.exists():
        raise FileExistsError(target)
    files = {path.relative_to(staged).as_posix(): sha256(path) for path in staged.rglob("*") if path.is_file()}
    if any(name not in files or (staged / name).stat().st_size == 0 for name in required):
        raise ValueError(f"Incomplete artifact: {staged}")
    write_json(staged / "complete.json", {"identity": identity, "step": step, "files": files})
    staged.rename(target)
    verify(target, identity, required)


def verify(path, identity, required):
    manifest = read_json(path / "complete.json")
    files = manifest["files"]
    if manifest["identity"] != identity or not set(required) <= files.keys():
        raise ValueError(f"Incompatible artifact: {path}")
    if any(sha256(path / name) != expected for name, expected in files.items()):
        raise ValueError(f"Checksum mismatch: {path}")
    return manifest["step"]


@contextmanager
def writer(run):
    run.mkdir(parents=True, exist_ok=True)
    lock = run / ".writer"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f"Run already locked. Inspect {lock / 'owner.json'} and see RESUME_INSTRUCTIONS.md.") from None
    owner = lock / "owner.json"
    try:
        write_json(owner, {"hostname": socket.gethostname(), "pid": os.getpid()})
        yield
    finally:
        if owner.exists():
            owner.unlink()
        lock.rmdir()


def archive(run, path):
    history = run / "history"
    history.mkdir(exist_ok=True)
    path.rename(history / f"{path.name}-{uuid4().hex}")
