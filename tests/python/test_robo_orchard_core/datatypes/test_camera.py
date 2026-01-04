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

import os

import cv2
import numpy as np
import pytest
import torch

from robo_orchard_core.datatypes.camera_data import (
    BatchCameraData,
    BatchCameraInfo,
    BatchFrameTransform,
    Distortion,
    ImageMode,
)
from robo_orchard_core.utils.math import math_utils
from robo_orchard_core.utils.math.transform import (
    Rotate2D,
    Scale2D,
    Transform2D_M,
    Translate2D,
)


@pytest.fixture(scope="session")
def img_lenna(workspace: str) -> torch.Tensor:
    """Fixture to load the Lenna image."""
    img_path = os.path.join(
        workspace, "robo_orchard_workspace", "imgs", "Lenna.png"
    )

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert isinstance(img, (np.ndarray,)), "Image not loaded correctly"
    return torch.asarray(img)  # torch.Size([500, 500, 3])


def get_affine_transform(
    center: tuple[float, float], angle: float, scale: float
) -> Transform2D_M:
    t = Translate2D([-center[0], -center[1]])
    r = Rotate2D(angle)
    s = Scale2D([scale, scale])
    return t @ r @ t.inverse() @ s


class TestBatchCameraData:
    def test_to_dict(self):
        a = BatchCameraData(
            sensor_data=torch.rand(size=(2, 12, 11, 3), dtype=torch.float32),
            pix_fmt=ImageMode.BGR,
            # with distortion
            distortion=Distortion(
                model="plumb_bob",
                coefficients=torch.tensor(
                    [0.1, 0.01, 0.001, 0.0001], dtype=torch.float32
                ),
            ),
        )
        d = a.model_dump()
        for field in BatchCameraData.model_fields:
            assert field in d, f"Field {field} is missing in the dumped dict"

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_apply_transform2d(self, img_lenna: torch.Tensor, batch_size: int):
        target_hw = (200, 200)
        src_hw = img_lenna.shape[:2]
        ts = get_affine_transform(
            center=(src_hw[1] / 2 + 4, src_hw[0] / 2 - 10),
            angle=np.deg2rad(45),
            scale=2.0 / 5.0,
        )
        sensor_data = img_lenna.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        intrinsic_matrices = torch.tensor(
            [
                [
                    [100, 0, src_hw[1] / 2],
                    [0, 100, src_hw[0] / 2],
                    [0, 0, 1],
                ]
            ]
            * batch_size,
            dtype=torch.float32,
        )
        data = BatchCameraData(
            sensor_data=sensor_data,
            intrinsic_matrices=intrinsic_matrices,
        )
        new_data = data.apply_transform2d(
            transform=ts,
            target_hw=target_hw,
        )
        assert data.intrinsic_matrices is not None
        gt_instrinsic_new = ts.get_matrix() @ data.intrinsic_matrices
        assert new_data.intrinsic_matrices is not None
        assert torch.allclose(
            gt_instrinsic_new, new_data.intrinsic_matrices, atol=1e-6
        )


class TestBatchCameraInfo:
    @pytest.fixture()
    def dummy_camera_info(
        self,
    ) -> BatchCameraInfo:
        intrinsic_matrices = torch.tensor(
            [
                [
                    [100, 0, 50],
                    [0, 100, 50],
                    [0, 0, 1],
                ],
                [
                    [200, 0, 100],
                    [0, 200, 100],
                    [0, 0, 1],
                ],
            ],
            dtype=torch.float32,
        )
        return BatchCameraInfo(
            intrinsic_matrices=intrinsic_matrices,
            frame_id="camera",
            pose=BatchFrameTransform(
                parent_frame_id="world",
                child_frame_id="camera",
                xyz=(torch.rand(size=(2, 3), dtype=torch.float32) - 0.5) * 10,
                quat=math_utils.normalize(
                    torch.rand(size=(2, 4), dtype=torch.double) - 0.5,
                    dim=-1,
                ),
            ),
        )

    @pytest.mark.parametrize(
        "frame_id, device",
        [
            ("camera", "cpu"),
            ("world", "cuda" if torch.cuda.is_available() else "cpu"),
        ],
    )
    def test_project_unproject_consistency(
        self, dummy_camera_info: BatchCameraInfo, frame_id: str, device: str
    ):
        batch_size = dummy_camera_info.intrinsic_matrices.shape[0]  # type: ignore
        num_points = 100
        points_3d = (
            torch.rand(size=(batch_size, num_points, 3), device=device) - 0.5
        ) * 10

        projected_uvd = dummy_camera_info.project_points_to_image(
            points_3d, frame_id=frame_id
        )
        unprojected_points_3d = dummy_camera_info.unproject_image_points(
            projected_uvd, frame_id=frame_id
        )

        assert torch.allclose(
            points_3d,
            unprojected_points_3d,
            atol=1e-5,
        ), "Unprojected points do not match original points"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
