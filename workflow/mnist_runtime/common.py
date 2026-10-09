from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path


class Blocked(RuntimeError):
    """An explicit condition prevents admission; never silently retry."""


class Conflict(Blocked):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def strict_json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise Blocked("invalid identifier")
    return value


def amount(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise Blocked("resource amounts must be finite nonnegative numbers")
    return value


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate(value, schema, schema_root):
    """Validate exactly the keywords used in the two frozen project schemas.

    Unknown assertion keywords fail closed; this is not a general JSON Schema engine.
    """
    known = {"$schema", "title", "description", "$ref", "type", "const", "enum",
             "properties", "required", "additionalProperties", "items"}
    if set(schema) - known:
        raise Blocked("unsupported schema keyword")
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref != "decision.schema.json":
            raise Blocked("unbound schema reference")
        return validate(value, strict_json((schema_root / ref).read_bytes()), schema_root)
    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        raise Blocked("schema constant mismatch")
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        raise Blocked("value outside enum")
    checks = {"null": value is None, "string": isinstance(value, str),
              "number": type(value) in (int, float) and math.isfinite(value),
              "object": isinstance(value, dict), "array": isinstance(value, list)}
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(checks.get(t, False) for t in types):
            raise Blocked("schema type mismatch")
    if isinstance(value, dict) and "properties" in schema:
        props = schema["properties"]
        if set(schema.get("required", [])) - set(value):
            raise Blocked("required schema field missing")
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise Blocked("extra schema field")
        for key, item in value.items():
            if key in props:
                validate(item, props[key], schema_root)
    if isinstance(value, list) and "items" in schema:
        for item in value:
            validate(item, schema["items"], schema_root)
