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
    BatchCameraDataEncoded,
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
        assert new_data.transform_matrices is not None
        assert torch.allclose(
            ts.get_matrix(), new_data.transform_matrices, atol=1e-6
        )

    def test_getitem_supports_int_slice_and_list(self):
        intrinsic_matrices = (
            torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        )
        data = BatchCameraData(
            sensor_data=torch.arange(3 * 2 * 2 * 1, dtype=torch.float32).view(
                3, 2, 2, 1
            ),
            pix_fmt=ImageMode.L,
            intrinsic_matrices=intrinsic_matrices,
            timestamps=[11, 22, 33],
            frame_id="camera",
        )

        data_int = data[1]
        assert data_int.batch_size == 1
        assert data_int.timestamps == [22]
        assert torch.equal(data_int.sensor_data, data.sensor_data[[1]])

        data_slice = data[1:]
        assert data_slice.batch_size == 2
        assert data_slice.timestamps == [22, 33]
        assert torch.equal(data_slice.sensor_data, data.sensor_data[1:])

        data_list = data[[2, 0]]
        assert data_list.batch_size == 2
        assert data_list.timestamps == [33, 11]
        assert torch.equal(data_list.sensor_data, data.sensor_data[[2, 0]])

    def test_to_keep_sensor_dtype_and_respect_dtype_exclude_fields(self):
        data = BatchCameraData(
            sensor_data=torch.randint(
                low=0,
                high=255,
                size=(2, 6, 5, 3),
                dtype=torch.uint8,
            ),
            pix_fmt=ImageMode.RGB,
            intrinsic_matrices=(
                torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
            ),
        )
        data.to(dtype=torch.float64)
        assert data.sensor_data.dtype == torch.uint8
        assert data.intrinsic_matrices is not None
        assert data.intrinsic_matrices.dtype == torch.float64

        data = BatchCameraData(
            sensor_data=torch.randint(
                low=0,
                high=255,
                size=(2, 6, 5, 3),
                dtype=torch.uint8,
            ),
            pix_fmt=ImageMode.RGB,
            intrinsic_matrices=(
                torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
            ),
        )
        data.to(
            dtype=torch.float64, dtype_exclude_fields=["intrinsic_matrices"]
        )
        assert data.sensor_data.dtype == torch.uint8
        assert data.intrinsic_matrices is not None
        assert data.intrinsic_matrices.dtype == torch.float32

    def test_encoded_getitem_supports_int_slice_and_list(self):
        encoded = BatchCameraDataEncoded(
            sensor_data=[b"a", b"b", b"c"],
            format="jpeg",
            intrinsic_matrices=(
                torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
            ),
            timestamps=[101, 102, 103],
            frame_id="camera",
        )

        encoded_int = encoded[2]
        assert encoded_int.sensor_data == [b"c"]
        assert encoded_int.timestamps == [103]

        encoded_slice = encoded[1:]
        assert encoded_slice.sensor_data == [b"b", b"c"]
        assert encoded_slice.timestamps == [102, 103]

        encoded_list = encoded[[2, 0]]
        assert encoded_list.sensor_data == [b"c", b"a"]
        assert encoded_list.timestamps == [103, 101]


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

    def test_getitem_and_get_intrinsic_with_transform(
        self, dummy_camera_info: BatchCameraInfo
    ):
        assert dummy_camera_info.intrinsic_matrices is not None
        transform_matrices = torch.tensor(
            [
                [[2.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]],
                [[1.5, 0.0, 1.0], [0.0, 1.5, 2.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        camera_info = dummy_camera_info.model_copy(
            update={"transform_matrices": transform_matrices}
        )

        selected = camera_info[[1, 0]]
        assert selected.intrinsic_matrices is not None
        assert selected.transform_matrices is not None
        assert selected.pose is not None
        assert selected.intrinsic_matrices.shape[0] == 2
        assert camera_info.intrinsic_matrices is not None
        assert torch.equal(
            selected.intrinsic_matrices, camera_info.intrinsic_matrices[[1, 0]]
        )
        assert torch.equal(
            selected.transform_matrices, transform_matrices[[1, 0]]
        )
        assert selected.pose.batch_size == 2

        intrinsic_with_transform = camera_info.get_intrinsic_with_transform()
        assert intrinsic_with_transform is not None
        assert torch.allclose(
            intrinsic_with_transform,
            torch.bmm(
                transform_matrices,
                dummy_camera_info.intrinsic_matrices,
            ),
            atol=1e-6,
        )

        no_transform = dummy_camera_info.model_copy(
            update={"transform_matrices": None}
        )
        intrinsic_clone = no_transform.get_intrinsic_with_transform()
        assert intrinsic_clone is not None
        assert no_transform.intrinsic_matrices is not None
        assert intrinsic_clone is not no_transform.intrinsic_matrices
        assert torch.equal(intrinsic_clone, no_transform.intrinsic_matrices)

    def test_concat_rejects_mixed_pose_presence(
        self, dummy_camera_info: BatchCameraInfo
    ):
        without_pose = dummy_camera_info.model_copy(update={"pose": None})

        with pytest.raises(ValueError, match="pose type"):
            BatchCameraInfo.concat([without_pose, dummy_camera_info])

    def test_concat_fills_missing_transform_matrices_per_batch(
        self, dummy_camera_info: BatchCameraInfo
    ):
        without_transform = dummy_camera_info.model_copy(
            update={"transform_matrices": None}
        )
        transform = torch.tensor(
            [[[2.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        with_transform = dummy_camera_info[0].model_copy(
            update={"transform_matrices": transform}
        )

        merged = BatchCameraInfo.concat([without_transform, with_transform])

        assert merged.transform_matrices is not None
        assert merged.transform_matrices.shape == (3, 3, 3)
        expected = torch.cat(
            [
                torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1),
                transform,
            ],
            dim=0,
        )
        assert torch.equal(merged.transform_matrices, expected)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
