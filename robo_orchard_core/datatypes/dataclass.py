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

"""Base data class and mixin."""

import copy
from typing import Any

import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from pydantic_core import from_json
from typing_extensions import Self

from robo_orchard_core.utils.config import (
    callable_to_string,
    string_to_callable,
)
from robo_orchard_core.utils.torch_utils import Device, make_device

__all__ = [
    "DataClass",
    "TensorToMixin",
    "tensor_equal",
    "np2torch",
]


class DataClass(BaseModel):
    """The base data class that extends pydantic's BaseModel.

    This class is used to define data classes that are used to store data
    and validate the data. It extends pydantic's BaseModel and adds a
    :py:meth:`__post_init__` method that can be used to perform additional
    initialization after the model is constructed.

    Note:
        Serialization and deserialization using pydantic's methods are not
        recommended for performance reasons, as data classes can be used to
        store large tensors or other data that are not easily serialized.

        User should implement the proper serialization and deserialization
        methods when needed.

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @model_serializer(mode="wrap", return_type=dict, when_used="always")
    def wrapped_model_ser(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ):
        """Serializes the configuration to a dictionary.

        This wrapper function is used when the configuration is serialized.
        It adds the `__class_type__` key to the dictionary.

        `__class_type__` is the string representation of the class type. It
        is used to determine the class type when deserializing the JSON string
        instead of using pydantic's default behavior.

        For builtin types, the `__class_type__` key will not be added to the
        dictionary.

        The `context` argument in the `model_dump` method is used to
        determine whether to include the `__class_type__` key in the
        serialized dictionary. If context['include_class_type'] is True,
        the `__class_type__` key will be added to the dictionary.

        """
        if self.__class__.__module__ == "builtins":
            return handler(self)
        if isinstance(info.context, dict) and info.context.get(
            "include_class_type", False
        ):
            ret = {"__class_type__": callable_to_string(type(self))}
            ret.update(handler(self))
            return ret
        else:
            return handler(self)

    @model_validator(mode="wrap")
    @classmethod
    def wrapped_model_val(
        cls, data: Any, handler: ValidatorFunctionWrapHandler
    ):
        if isinstance(data, str):
            data = from_json(data, allow_partial=True)
        if isinstance(data, dict):
            if "__class_type__" in data:
                data = data.copy()
                target_cls = string_to_callable(data.pop("__class_type__"))
                if target_cls == cls:
                    return handler(data)
                else:
                    return target_cls.model_validate(data)
            else:
                return handler(data)
        return data

    def __post_init__(self):
        """Hack to replace __post_init__ in configclass."""
        pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value of the feature by key.

        This method provides a dict like get method to access the
        attributes of the data class.
        """
        return getattr(self, key, default)

    def model_post_init(self, *args, **kwargs):
        """Post init method for the model.

        Perform additional initialization after :py:meth:`__init__`
        and model_construct. This is useful if you want to do some validation
        that requires the entire model to be initialized.

        To be consistent with configclass, this method is implemented by
        calling the :py:meth:`__post_init__` method.

        """
        self.__post_init__()

    def _tensor_to_field_items(self) -> list[tuple[str, Any]]:
        """Return mutable field items for :meth:`TensorToMixin.to`."""
        return [
            (field_name, getattr(self, field_name))
            for field_name in type(self).model_fields
        ]

    def _tensor_to_copy_with_updates(
        self, updated_fields: dict[str, Any]
    ) -> Self:
        """Build a shallow updated copy for :meth:`TensorToMixin.to`."""
        return self.model_copy(update=updated_fields, deep=False)

    def __eq__(self, other: Any) -> bool:
        # use the default equality method first.
        try:
            return super().__eq__(other)
        except Exception:
            pass

        def obj_eq(src, dst):
            """Custom equality method for objects with tensor support."""
            if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
                ret = tensor_equal(src, dst)
                return ret
            elif isinstance(src, (list, tuple)) and isinstance(
                dst, (list, tuple)
            ):
                if len(src) != len(dst):
                    return False
                return all(obj_eq(s, d) for s, d in zip(src, dst, strict=True))
            elif isinstance(src, dict) and isinstance(dst, dict):
                if src.keys() != dst.keys():
                    return False
                return all(obj_eq(src[k], dst[k]) for k in src.keys())
            else:
                return src == dst

        # if the default equality method fails, we use the custom
        # equality method that compares the __dict__ attributes of the objects.
        if not isinstance(other, DataClass):
            raise NotImplementedError()
        return obj_eq(self.__dict__, other.__dict__)

    @classmethod
    def dataset_feature(cls, *args, **kwargs):
        """A decorator to get huggingface dataset feature.

        This method provide a placeholder for getting the corresponding
        huggingface dataset feature for the data class.
        """
        raise NotImplementedError(
            "dataset_feature is not implemented for DataClass."
            "Please implement it in the subclass."
        )


class TensorToMixin:
    def _tensor_to_field_items(self) -> list[tuple[str, Any]]:
        """Return attribute items that :meth:`to` should traverse.

        Classes with non-field attributes, caches, or framework-specific
        storage can override this hook to expose only the mutable payload that
        should participate in backend alignment.
        """
        try:
            return list(vars(self).items())
        except TypeError as exc:
            raise TypeError(
                f"{type(self).__name__}.to() requires either __dict__-backed "
                "attributes or a custom _tensor_to_field_items() override."
            ) from exc

    def _tensor_to_copy_with_updates(
        self, updated_fields: dict[str, Any]
    ) -> Self:
        """Build a shallow updated copy for :meth:`to`."""
        copied_self = copy.copy(self)
        for field_name, updated_value in updated_fields.items():
            setattr(copied_self, field_name, updated_value)
        return copied_self

    def to(
        self,
        device: Device | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        dtype_exclude_fields: list[str] | None = None,
        inplace: bool = False,
    ) -> Self:
        """Move or cast the tensors/modules in the data class.

        By default this method returns a new aligned object when a backend
        change is required and returns ``self`` on a no-op request. Passing
        ``inplace=True`` preserves the container identity and writes the
        converted field values back onto ``self``.

        Args:
            device (Device|None, optional): The target device to move the
                tensors/modules to. If None, the device will not be changed.
                Defaults to None.
            dtype (torch.dtype | None, optional): The target dtype to cast
                the tensors to. If None, the dtype will not be changed.
            non_blocking (bool, optional): If True, the operation will be
                performed in a non-blocking manner. Defaults to False.
            dtype_exclude_fields (list[str] | None, optional): A list of
                field names to exclude from dtype conversion. If None, all
                fields will be converted.
            inplace (bool, optional): If True, preserve the current container
                identity and assign converted field values back onto
                ``self``. Defaults to False.

        Returns:
            Self: The aligned data object.
        """

        device = make_device(device) if device is not None else None
        updated_fields: dict[str, Any] = {}
        any_changed = False
        for k, obj in self._tensor_to_field_items():
            if dtype_exclude_fields is not None and k in dtype_exclude_fields:
                updated_obj, changed = apply_to(
                    obj,
                    device=device,
                    dtype=None,
                    non_blocking=non_blocking,
                )
            else:
                updated_obj, changed = apply_to(
                    obj,
                    device=device,
                    dtype=dtype,
                    non_blocking=non_blocking,
                )

            updated_fields[k] = updated_obj
            any_changed = any_changed or changed

        if not any_changed:
            return self

        if inplace:
            for k, updated_obj in updated_fields.items():
                setattr(self, k, updated_obj)
            return self

        return self._tensor_to_copy_with_updates(updated_fields)


def tensor_equal(
    src: torch.Tensor | None,
    dst: torch.Tensor | None,
    rtol: float = 0.00001,
    atol: float = 1e-8,
) -> bool:
    """Check if two tensors are equal within a tolerance.

    Args:
        src (torch.Tensor | None): The first tensor.
        dst (torch.Tensor | None): The second tensor.
        eps (float, optional): The tolerance for equality. Defaults to 1e-6.

    Returns:
        bool: True if the tensors are equal within the tolerance,
            False otherwise.
    """
    if [src, dst].count(None) == 1:
        return False
    if src is None and dst is None:
        return True
    assert src is not None and dst is not None
    return torch.allclose(src, dst, rtol=rtol, atol=atol)


def np2torch(src: np.ndarray | list | dict | torch.Tensor) -> Any:
    """Convert numpy array to torch tensor."""
    if isinstance(src, torch.Tensor):
        return src
    if isinstance(src, np.ndarray):
        return torch.from_numpy(src)
    elif isinstance(src, (list, tuple)):
        return [np2torch(obj) for obj in src]
    elif isinstance(src, dict):
        return {k: np2torch(v) for k, v in src.items()}
    else:
        return src


def _tensor_requires_to(
    tensor: torch.Tensor,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> bool:
    return (device is not None and tensor.device != device) or (
        dtype is not None and tensor.dtype != dtype
    )


def _module_requires_to(
    module: torch.nn.Module,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> bool:
    if device is None and dtype is None:
        return False

    tensors = [*module.parameters(), *module.buffers()]
    return any(
        _tensor_requires_to(tensor, device=device, dtype=dtype)
        for tensor in tensors
    )


def apply_to(
    obj: Any,
    device: torch.device | None,
    dtype: torch.dtype | None,
    non_blocking: bool,
) -> tuple[Any, bool]:
    if isinstance(obj, torch.Tensor):
        if not _tensor_requires_to(obj, device=device, dtype=dtype):
            return obj, False
        return (
            obj.to(device=device, dtype=dtype, non_blocking=non_blocking),
            True,
        )
    elif isinstance(obj, torch.nn.Module):
        if not _module_requires_to(obj, device=device, dtype=dtype):
            return obj, False
        aligned_module = copy.deepcopy(obj)
        aligned_module.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        )
        return aligned_module, True
    elif isinstance(obj, TensorToMixin):
        aligned_obj = obj.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            inplace=False,
        )
        return aligned_obj, aligned_obj is not obj
    elif isinstance(obj, list):
        updated_list = []
        any_changed = False
        for item in obj:
            updated_item, changed = apply_to(
                item,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            updated_list.append(updated_item)
            any_changed = any_changed or changed
        if not any_changed:
            return obj, False
        return updated_list, True
    elif isinstance(obj, tuple):
        updated_items = []
        any_changed = False
        for item in obj:
            updated_item, changed = apply_to(
                item,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            updated_items.append(updated_item)
            any_changed = any_changed or changed
        if not any_changed:
            return obj, False
        return tuple(updated_items), True
    elif isinstance(obj, dict):
        updated_dict = {}
        any_changed = False
        for k, v in obj.items():
            updated_value, changed = apply_to(
                v,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            updated_dict[k] = updated_value
            any_changed = any_changed or changed
        if not any_changed:
            return obj, False
        return updated_dict, True
    else:
        return obj, False
