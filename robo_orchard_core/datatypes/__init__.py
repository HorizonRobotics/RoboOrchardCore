# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

"""Tensor-backed datatype entrypoints, loaded on demand."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robo_orchard_core.datatypes.camera_data import (
        BatchCameraData,
        BatchCameraDataEncoded,
        BatchCameraInfo,
        BatchImageData,
        DecoderType,
        Distortion,
        EncoderType,
        ImageChannelLayout,
        ImageMode,
    )
    from robo_orchard_core.datatypes.dataclass import (
        DataClass,
        TensorToMixin,
        np2torch,
        tensor_equal,
    )
    from robo_orchard_core.datatypes.geometry import (
        BatchFrameTransform,
        BatchPose,
        BatchPose6D,
        BatchTransform3D,
    )
    from robo_orchard_core.datatypes.joint_state import BatchJointsState
    from robo_orchard_core.datatypes.tf_graph import (
        BatchFrameTransformGraph,
        BatchFrameTransformGraphState,
    )
    from robo_orchard_core.utils.config import NumpyTensor, TorchTensor

__all__ = [
    "EncoderType",
    "DecoderType",
    "ImageChannelLayout",
    "ImageMode",
    "Distortion",
    "BatchCameraInfo",
    "BatchImageData",
    "BatchCameraData",
    "BatchCameraDataEncoded",
    "DataClass",
    "TensorToMixin",
    "tensor_equal",
    "np2torch",
    "BatchTransform3D",
    "BatchPose6D",
    "BatchPose",
    "BatchFrameTransform",
    "BatchJointsState",
    "BatchFrameTransformGraphState",
    "BatchFrameTransformGraph",
    "TorchTensor",
    "NumpyTensor",
]

_EXPORTS = (
    {
        name: ("camera_data", name)
        for name in (
            "EncoderType",
            "DecoderType",
            "ImageChannelLayout",
            "ImageMode",
            "Distortion",
            "BatchCameraInfo",
            "BatchImageData",
            "BatchCameraData",
            "BatchCameraDataEncoded",
        )
    }
    | {
        name: ("dataclass", name)
        for name in ("DataClass", "TensorToMixin", "tensor_equal", "np2torch")
    }
    | {
        name: ("geometry", name)
        for name in (
            "BatchTransform3D",
            "BatchPose6D",
            "BatchPose",
            "BatchFrameTransform",
        )
    }
    | {"BatchJointsState": ("joint_state", "BatchJointsState")}
    | {
        name: ("tf_graph", name)
        for name in (
            "BatchFrameTransformGraphState",
            "BatchFrameTransformGraph",
        )
    }
)
_ROBOTICS_MODULES = frozenset(
    {
        "torch",
        "numpy",
        "numpydantic",
        "PIL",
        "cv2",
    }
)
_OPTIONAL_EXPORT_MODULES = {
    "TorchTensor": "robo_orchard_core.utils.config",
    "NumpyTensor": "robo_orchard_core.utils.config",
}


def __getattr__(name: str) -> Any:
    """Resolve a datatype without importing robotics modules early."""
    if name in _OPTIONAL_EXPORT_MODULES:
        value = getattr(
            import_module(_OPTIONAL_EXPORT_MODULES[name]),
            name,
        )
        globals()[name] = value
        return value

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
