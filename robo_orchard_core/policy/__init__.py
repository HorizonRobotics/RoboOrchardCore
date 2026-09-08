# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Policy entrypoints, loaded on demand with the robotics runtime."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robo_orchard_core.policy.base import (
        ClassType,
        PolicyConfig,
        PolicyConfigType_co,
        PolicyMixin,
    )
    from robo_orchard_core.policy.random import (
        RandomPolicy,
        RandomPolicyConfig,
    )

__all__ = [
    "PolicyMixin",
    "PolicyConfig",
    "ClassType",
    "PolicyConfigType_co",
    "RandomPolicy",
    "RandomPolicyConfig",
]

_EXPORTS = {
    name: ("base", name)
    for name in (
        "PolicyMixin",
        "PolicyConfig",
        "ClassType",
        "PolicyConfigType_co",
    )
} | {name: ("random", name) for name in ("RandomPolicy", "RandomPolicyConfig")}
_ROBOTICS_MODULES = frozenset({"torch", "gymnasium", "numpy", "numpydantic"})


def __getattr__(name: str) -> Any:
    """Resolve a supported policy without importing robotics modules early."""
    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error

    try:
        module = import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as error:
        module_name = (error.name or "").split(".", maxsplit=1)[0]
        if module_name not in _ROBOTICS_MODULES:
            raise
        raise ModuleNotFoundError(
            f"{__name__}.{name} requires the robotics runtime. Install "
            "'robo_orchard_core[robotics]'.",
            name=error.name,
        ) from None

    value = getattr(module, symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to interactive tools without importing them."""
    return sorted({*globals(), *__all__})
