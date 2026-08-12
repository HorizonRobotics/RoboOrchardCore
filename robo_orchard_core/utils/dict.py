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

import json
from collections import OrderedDict
from typing import Any, Literal

__all__ = ["flatten_dict"]


def flatten_dict(
    d: dict | list | tuple,
    parent_key: str = "",
    sep: str = "/",
    keep_order: bool = False,
    sequence_mode: Literal["flatten", "json"] = "flatten",
) -> dict[str, Any]:
    """Flatten nested mappings and sequences into a flat dictionary.

    Nested dictionaries are always flattened. By default, nested lists and
    tuples are also flattened into ``index_<n>`` keys. Set
    ``sequence_mode="json"`` to keep each nested sequence as one leaf value
    serialized to compact JSON; the JSON encoder recursively handles any
    mappings or sequences inside that sequence.

    For example, given the following dictionary:

    .. code-block: text

        {
            "a": 1,
            "b": {
                "c": 2,
                "d": [3, 4],
            }
        }

    The flattened dictionary would be:

    .. code-block: text

        {
            "a": 1,
            "b/c": 2,
            "b/d/index_0": 3,
            "b/d/index_1": 4,
        }

    Args:
        d (dict | list | tuple): The dictionary or sequence to flatten.
        parent_key (str, optional): Parent key prepended to flattened keys.
            Default is ``""``.
        sep (str, optional): Separator between flattened key components.
            Default is ``"/"``.
        keep_order (bool, optional): Whether to return an ordered mapping.
            Default is ``False``.
        sequence_mode (Literal["flatten", "json"], optional): How nested
            list and tuple values are represented. ``"flatten"`` recursively
            expands them into indexed keys; ``"json"`` serializes each
            complete sequence subtree as compact JSON. Default is
            ``"flatten"``.

    Returns:
        dict[str, Any]: The flattened dictionary.

    Raises:
        TypeError: If ``d`` is not a dictionary, list, or tuple, or if a
            sequence selected for JSON serialization contains unsupported
            values.
        ValueError: If ``sequence_mode`` is unsupported.
    """

    if sequence_mode not in ("flatten", "json"):
        raise ValueError(
            "sequence_mode must be either 'flatten' or 'json', "
            f"got {sequence_mode!r}."
        )

    items = []
    if isinstance(d, dict):
        item_iter = d.items()
    elif isinstance(d, (list, tuple)):
        item_iter = enumerate(d)
    else:
        raise TypeError(
            "Input argument `d` should be a dictionary, list, or tuple. "
            f"Received: {d}"
        )

    for k, v in item_iter:
        if isinstance(k, int):
            k = f"index_{k}"
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) or (
            sequence_mode == "flatten" and isinstance(v, (list, tuple))
        ):
            items.extend(
                flatten_dict(
                    v,
                    new_key,
                    sep=sep,
                    keep_order=keep_order,
                    sequence_mode=sequence_mode,
                ).items()
            )
        elif isinstance(v, (list, tuple)):
            try:
                serialized_value = json.dumps(
                    v,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Sequence at key {new_key!r} is not JSON serializable: "
                    f"{exc}"
                ) from exc
            items.append((new_key, serialized_value))
        else:
            items.append((new_key, v))
    if keep_order:
        return OrderedDict(items)
    return dict(items)
