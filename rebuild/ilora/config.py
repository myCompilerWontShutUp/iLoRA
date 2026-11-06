"""M0 설정 (DESIGN.md §4.6.2).

A profile is a YAML file whose top-level keys are config sections. Every key
of every section is required: there are no code defaults for anything that
changes the meaning of an experiment. A profile may start from another one
with ``extends: <other.yaml>`` (path relative to the profile), and command
line overrides use ``--set section.key=value``.

Each key is declared with ``item(label, doc)``; the label is the evidence
label of DESIGN.md §0.1 and is checked by the tests.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

LABEL_RE = re.compile(r"^\[(논문|v1|LLaRA|LLaRA 간접|코드|결정|SPEC)[^\]]*\]")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class ConfigError(ValueError):
    """Invalid profile: unknown key, missing key, wrong type or value."""


def item(label: str, doc: str, *, choices: Optional[Sequence[Any]] = None) -> Any:
    """Declare one required config key with its evidence label."""
    if not LABEL_RE.match(label):
        raise ValueError(f"invalid evidence label {label!r}")
    return field(metadata={"label": label, "doc": doc, "choices": tuple(choices) if choices else None})


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _deep_merge(base: dict, over: Mapping) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def load_raw(path: Path, _seen: Optional[set] = None) -> dict:
    """Read a profile, resolving ``extends`` chains (cycles are errors)."""
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ConfigError(f"extends cycle at {path}")
    seen.add(path)
    data = _read_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str):
        raise ConfigError(f"{path}: extends must be a file name")
    return _deep_merge(load_raw(path.parent / parent, seen), data)


def parse_override(text: str) -> tuple[str, Any]:
    """``section.key=value`` -> (``section.key``, YAML-parsed value)."""
    if "=" not in text:
        raise ConfigError(f"override must look like section.key=value: {text!r}")
    key, value = text.split("=", 1)
    key = key.strip()
    if key.count(".") != 1:
        raise ConfigError(f"override key must be section.key: {key!r}")
    return key, yaml.safe_load(value)


def _coerce(value: Any, tp: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        if len(inner) != 1:
            raise ConfigError(f"{where}: unsupported type {tp}")
        return _coerce(value, inner[0], where)
    if origin in (tuple, list) or tp in (tuple, list):
        args = typing.get_args(tp)
        elem = args[0] if args else Any
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        return tuple(_coerce(v, elem, f"{where}[{i}]") for i, v in enumerate(value))
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        # PyYAML (YAML 1.1) reads "1e-4" as a string; accept plain numeric text.
        if isinstance(value, str) and _FLOAT_RE.match(value.strip()):
            return float(value)
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    if tp is Any:
        return value
    raise ConfigError(f"{where}: unsupported type {tp}")


def _build_section(cls: type, raw: Any, name: str) -> Any:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"section {name!r} must be a mapping")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown key(s) in section {name!r}: {unknown}")
    missing = sorted(known - set(raw))
    if missing:
        raise ConfigError(f"missing key(s) in section {name!r}: {missing}")
    values = {}
    for f in fields(cls):
        where = f"{name}.{f.name}"
        value = _coerce(raw[f.name], hints[f.name], where)
        choices = f.metadata.get("choices")
        if choices is not None and value not in choices:
            raise ConfigError(f"{where}: {value!r} is not one of {list(choices)}")
        values[f.name] = value
    obj = cls(**values)
    check = getattr(obj, "validate", None)
    if callable(check):
        check()
    return obj


def build_config(raw: Mapping, schema: type = None) -> Any:
    """Validate a raw mapping against the schema and build frozen sections."""
    schema = schema or _default_schema()
    hints = typing.get_type_hints(schema)
    known = {f.name for f in fields(schema)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown section(s): {unknown}")
    missing = sorted(known - set(raw))
    if missing:
        raise ConfigError(f"missing section(s): {missing}")
    sections = {f.name: _build_section(hints[f.name], raw[f.name], f.name) for f in fields(schema)}
    cfg = schema(**sections)
    check = getattr(cfg, "validate", None)
    if callable(check):
        check()
    return cfg


def load_config(path: Path, overrides: Sequence[str] = (), schema: type = None) -> tuple[Any, dict]:
    """Load a profile and apply ``--set`` overrides.

    Returns the frozen config and the applied overrides (for meta.json).
    """
    raw = load_raw(Path(path))
    applied: dict[str, Any] = {}
    for text in overrides:
        key, value = parse_override(text)
        section, name = key.split(".")
        if section not in raw or not isinstance(raw[section], dict):
            raise ConfigError(f"override for unknown section {section!r}")
        if name not in raw[section]:
            raise ConfigError(f"override for unknown key {key!r}")
        raw[section][name] = value
        applied[key] = value
    return build_config(raw, schema), applied


# ----------------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------------
def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def config_to_dict(cfg: Any) -> dict:
    out = {}
    for f in fields(cfg):
        section = getattr(cfg, f.name)
        out[f.name] = {g.name: _plain(getattr(section, g.name)) for g in fields(section)}
    return out


def dump_config(cfg: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, allow_unicode=True, sort_keys=False)


def config_hash(cfg: Any) -> str:
    text = json.dumps(config_to_dict(cfg), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_items(schema: type = None):
    """Yield (section, key, label, doc) for every declared key."""
    schema = schema or _default_schema()
    hints = typing.get_type_hints(schema)
    for f in fields(schema):
        for g in fields(hints[f.name]):
            yield f.name, g.name, g.metadata.get("label"), g.metadata.get("doc")


def _default_schema() -> type:
    # Sections are declared in sections.py (one dataclass per module section).
    from .sections import Config

    return Config


__all__ = [
    "ConfigError", "item", "load_raw", "parse_override", "build_config", "load_config",
    "config_to_dict", "dump_config", "config_hash", "iter_items", "LABEL_RE",
]
