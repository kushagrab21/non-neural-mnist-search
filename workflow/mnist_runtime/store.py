from __future__ import annotations

import fcntl
import os
from pathlib import Path

from .common import Blocked, Conflict, atomic_write, canonical, digest, identifier, now, strict_json


class Store:
    """Single-process writer capability. Directory modes are not an OS role sandbox."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = None
        for name in ("coordination", "archive", "reports", "invocations"):
            (self.root / name).mkdir(exist_ok=True)
        for role in ("runner", "researcher", "controller", "maintainer"):
            (self.root / "workspaces" / role / "outbox").mkdir(parents=True, exist_ok=True)
        self.path = self.root / "coordination/events.jsonl"
        self.events = []

    def __enter__(self):
        self.lock = (self.root / "coordination/writer.lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.replay()
        except Exception:
            self.lock.close()
            self.lock = None
            raise
        return self

    def __exit__(self, *args):
        if self.lock:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None

    def replay(self):
        self.events = []
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise Conflict("incomplete event tail retained; explicit recovery required")
        previous = "0" * 64
        seen = set()
        for line in raw.splitlines():
            try:
                event = strict_json(line)
                expected = event.pop("hash")
                if event["seq"] != len(self.events) + 1 or event["previous_hash"] != previous:
                    raise ValueError("event chain")
                if digest(event) != expected or event["event_id"] in seen:
                    raise ValueError("event hash/identity")
                event["hash"] = expected
                seen.add(event["event_id"])
                previous = expected
                self.events.append(event)
            except (ValueError, KeyError, TypeError) as exc:
                raise Conflict("corrupt event log; original bytes retained") from exc
        return self.events

    def find(self, event_id):
        return next((e for e in self.events if e["event_id"] == event_id), None)

    def append(self, event_id, kind, actor, payload, *, event_time=None, source=None):
        if self.lock is None:
            raise Blocked("writer lock required")
        identifier(event_id)
        content = {"event_id": event_id, "kind": kind, "actor": actor,
                   "payload": payload, "event_time": event_time, "source": source}
        # Canonicalization validates finite values before any mutation.
        content = strict_json(canonical(content))
        existing = self.find(event_id)
        if existing:
            if any(existing[k] != v for k, v in content.items()):
                raise Conflict("same event ID with different contents: " + event_id)
            return existing
        event = dict(content, schema_version="runtime-event/1", seq=len(self.events) + 1,
                     previous_hash=self.events[-1]["hash"] if self.events else "0" * 64,
                     receipt_time=now())
        event["hash"] = digest(event)
        with self.path.open("ab") as handle:
            handle.write(canonical(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(event)
        return event

    def archive(self, data):
        if self.lock is None:
            raise Blocked("writer lock required")
        if not isinstance(data, bytes) or len(data) > 16 * 1024 * 1024:
            raise Blocked("artifact must be bytes within 16 MiB preparation limit")
        sha = digest(data)
        path = self.root / "archive" / sha
        try:
            with path.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            path.chmod(0o444)
        except FileExistsError:
            if path.is_symlink() or digest(path.read_bytes()) != sha:
                raise Conflict("content-address archive corrupted")
        return sha

    def read_artifact(self, sha):
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise Blocked("invalid artifact hash")
        path = self.root / "archive" / sha
        if path.is_symlink() or digest(path.read_bytes()) != sha:
            raise Conflict("artifact identity mismatch")
        return path.read_bytes()

    def write_outbox(self, bound_role, relative, data):
        if bound_role not in {"runner", "researcher", "controller", "maintainer"}:
            raise Blocked("unknown bound role")
        base = self.root / "workspaces" / bound_role / "outbox"
        rel = Path(relative)
        if rel.is_absolute() or not rel.parts or any(p in ("..", ".") for p in rel.parts):
            raise Blocked("outbox path escape")
        target = base / rel
        if not target.resolve().is_relative_to(base.resolve()):
            raise Blocked("outbox symlink escape")
        for ancestor in [target, *target.parents]:
            if ancestor == self.root:
                break
            if ancestor.is_symlink():
                raise Blocked("outbox symlink forbidden")
        if target.exists():
            if target.read_bytes() != data:
                raise Conflict("outbox submission already frozen")
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(data)
        return target

    def write_index(self, state):
        if self.lock is None:
            raise Blocked("writer lock required")
        atomic_write(self.root / "coordination/index.json", canonical(state) + b"\n")
