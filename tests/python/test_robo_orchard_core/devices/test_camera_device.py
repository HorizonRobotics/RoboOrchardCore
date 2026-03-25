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

import torch

from robo_orchard_core.datatypes.geometry import BatchFrameTransform
from robo_orchard_core.devices.cameras.camera import CameraBase
from robo_orchard_core.utils.math import math_utils


class DummyCamera(CameraBase):
    @property
    def image_shape(self) -> tuple[int, int]:
        return (5, 6)

    @property
    def intrinsic_matrix(self) -> torch.Tensor:
        return torch.eye(3, dtype=torch.float32)

    @property
    def pose_global(self) -> BatchFrameTransform:
        return BatchFrameTransform(
            parent_frame_id="world",
            child_frame_id="camera",
            xyz=torch.zeros((1, 3), dtype=torch.float32),
            quat=math_utils.normalize(
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.double),
                dim=-1,
            ),
        )

    @property
    def sensor_data(self) -> torch.Tensor:
        return torch.zeros((5, 6, 3), dtype=torch.uint8)


class TestCameraBase:
    def test_get_camera_data_batches_single_camera_outputs(self):
        camera = DummyCamera.__new__(DummyCamera)
        camera.cfg = None

        data = camera.get_camera_data()

        assert data.sensor_data.shape == (1, 5, 6, 3)
        assert data.intrinsic_matrices is not None
        assert data.intrinsic_matrices.shape == (1, 3, 3)
        assert data.pose is not None
        assert data.pose.batch_size == 1
        assert data.image_shape == (5, 6)
