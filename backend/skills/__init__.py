"""Skill registry. Import this package to auto-register all skills in it."""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Skill:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]


_REGISTRY: dict[str, Skill] = {}


def skill(name: str, description: str, parameters: dict[str, Any]) -> Callable:
    """Decorator that registers a function as a skill callable by the agent.

    `parameters` is a JSON Schema object describing the function arguments.
    The handler receives kwargs matching the schema and a `session` kwarg
    carrying the per-chat session (uploaded folder path, etc.).
    """
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = Skill(name=name, description=description,
                                parameters=parameters, handler=fn)
        return fn
    return wrap


def all_skills() -> list[Skill]:
    return list(_REGISTRY.values())


def get(name: str) -> Skill | None:
    return _REGISTRY.get(name)


def load_all() -> None:
    """Import every submodule so their @skill decorators run."""
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{mod.name}")
