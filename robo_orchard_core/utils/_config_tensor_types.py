# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

"""Optional Tensor aliases used by :mod:`robo_orchard_core.utils.config`."""

from typing import Annotated

import torch
from numpydantic import NDArray
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import PlainValidator

__all__ = ["TorchTensor", "NumpyTensor"]


TorchTensor = Annotated[
    torch.Tensor,
    PlainValidator(
        lambda x: torch.tensor(x) if not isinstance(x, torch.Tensor) else x
    ),
    PlainSerializer(lambda x: x.tolist(), return_type=list, when_used="json"),
]

NumpyTensor = NDArray
