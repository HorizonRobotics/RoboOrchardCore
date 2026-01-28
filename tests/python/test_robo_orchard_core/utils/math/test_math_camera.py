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

from typing import Literal

import cv2
import numpy as np
import pytest
import torch

from robo_orchard_core.datatypes import BatchCameraInfo, BatchFrameTransform
from robo_orchard_core.utils.math import math_utils, transform3d
from robo_orchard_core.utils.math.camera import (
    project_points_to_image,
    unproject_image_points,
)


class TestCamera:
    def test_cv2_consistency(self):
        batch_size = 8
        num_points = 5
        device = "cpu"
        # Random 3D points
        points_3d = (
            torch.rand(
                size=(batch_size, num_points, 3),
                device=device,
                dtype=torch.double,
            )
            - 0.5
        ) * 100.0

        # Random intrinsic matrix
        fx = (
            torch.rand(batch_size, device=device, dtype=torch.double) * 800
            + 200
        )
        fy = (
            torch.rand(batch_size, device=device, dtype=torch.double) * 800
            + 200
        )
        cx = torch.rand(batch_size, device=device, dtype=torch.double) * 640
        cy = torch.rand(batch_size, device=device, dtype=torch.double) * 480

        intrinsic_mats = torch.zeros(
            batch_size, 3, 3, device=device, dtype=torch.double
        )
        intrinsic_mats[:, 0, 0] = fx
        intrinsic_mats[:, 1, 1] = fy
        intrinsic_mats[:, 0, 2] = cx
        intrinsic_mats[:, 1, 2] = cy
        intrinsic_mats[:, 2, 2] = 1.0

        trans_v = (
            torch.rand(size=(batch_size, 3), device=device, dtype=torch.double)
            - 0.5
        ) * 100.0

        rot_q = math_utils.normalize(
            torch.rand(size=(batch_size, 4), device=device, dtype=torch.double)
            - 0.5,
            dim=-1,
        )
        rot_v = math_utils.quaternion_to_axis_angle(rot_q)

        extrinsic_mats = transform3d.Transform3D_M.from_quat_trans(
            Q=rot_q, T=trans_v
        ).get_matrix()[:, :3, :4]

        projection_mats = intrinsic_mats @ extrinsic_mats

        # Project points to image
        projected_points = project_points_to_image(
            points_3d, projection_mats
        )  # (B, P, 3)

        # cv2 projection
        projected_points_cv2 = []
        for b in range(batch_size):
            points = points_3d[b].cpu().numpy().astype(np.float64)
            rvec = rot_v[b].cpu().numpy().astype(np.float64)
            tvec = trans_v[b].cpu().numpy().astype(np.float64)
            K = intrinsic_mats[b].cpu().numpy().astype(np.float64)
            img_points, _ = cv2.projectPoints(
                objectPoints=points,
                rvec=rvec,
                tvec=tvec,
                cameraMatrix=K,
                distCoeffs=np.array([0, 0, 0, 0, 0], dtype=np.float64),
            )
            projected_points_cv2.append(img_points.reshape(-1, 2))
        projected_points_cv2 = torch.from_numpy(
            np.stack(projected_points_cv2, axis=0)
        ).to(device)

        assert torch.allclose(
            projected_points[..., :2],
            projected_points_cv2,
            # atol=1e-2,
            # rtol=1e-4,
        ), "Projected points do not match cv2 results"

        # test BatchCameraInfo projection
        cam_info = BatchCameraInfo(
            intrinsic_matrices=intrinsic_mats,
            pose=BatchFrameTransform(
                xyz=trans_v,
                quat=rot_q,
                parent_frame_id="camera",
                child_frame_id="world",
            ).inverse(),
            frame_id="camera",
        )
        cam_projected_points = cam_info.project_points_to_image(
            points_3d, frame_id="world"
        )
        assert torch.allclose(
            projected_points,
            cam_projected_points,
            atol=1e-6,
            rtol=1e-4,
        ), "Projected points do not match BatchCameraInfo projection results"

    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA NOT AVAILABLE"
                ),
            ),
        ],
    )
    def test_project_unproject_consistency(self, device: str):
        def _gen_points(
            extrinsic_mats: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: E501
            # Define a safety threshold for depth.
            # Points closer than this to the camera plane are
            # numerically unstable.
            EPSILON_DEPTH = 0.1

            # Random 3D points
            points_3d = (
                torch.rand(size=(batch_size, num_points, 3), device=device)
                - 0.5  # noqa: E501
            ) * 100.0

            # Calculate the depth of each point in the Camera Coordinate
            # System.
            # P_cam = Extrinsic * P_world
            # We need homogeneous coordinates
            # for P_world to multiply with 3x4 Extrinsic
            ones = torch.ones((batch_size, num_points, 1), device=device)
            points_homo = torch.cat([points_3d, ones], dim=-1)  # (B, P, 4)

            # Matrix Multiplication: (B, 3, 4) @ (B, 4, P) -> (B, 3, P)
            points_cam = torch.bmm(extrinsic_mats, points_homo.transpose(1, 2))

            # Extract Z-depth (Batch, Num_Points)
            # Assuming standard convention where Z is depth
            depths = points_cam[:, 2, :]

            # Create a mask for valid points
            # (in front of camera and not too close)
            valid_mask = depths > EPSILON_DEPTH

            # Sanity check: Ensure we actually have some valid points to test
            num_valid = valid_mask.sum()
            if num_valid == 0:
                pytest.skip(
                    "Skipping test: No random points generated in front of camera."  # noqa: E501
                )

            return points_3d, valid_mask

        batch_size = 6
        num_points = 100

        # Random intrinsic matrix
        fx = torch.rand(batch_size, device=device) * 800 + 200
        fy = torch.rand(batch_size, device=device) * 800 + 200
        cx = torch.rand(batch_size, device=device) * 640
        cy = torch.rand(batch_size, device=device) * 480

        intrinsic_mats = torch.zeros(batch_size, 3, 3, device=device)
        intrinsic_mats[:, 0, 0] = fx
        intrinsic_mats[:, 1, 1] = fy
        intrinsic_mats[:, 0, 2] = cx
        intrinsic_mats[:, 1, 2] = cy
        intrinsic_mats[:, 2, 2] = 1.0

        trans_v = (
            torch.rand(size=(batch_size, 3), device=device) - 0.5
        ) * 100.0

        rot_q = math_utils.normalize(
            torch.rand(size=(batch_size, 4), device=device) - 0.5, dim=-1
        )
        extrinsic_mats = transform3d.Transform3D_M.from_quat_trans(
            Q=rot_q, T=trans_v
        ).get_matrix()[:, :3, :4]

        projection_mats = intrinsic_mats @ extrinsic_mats

        # Random points
        points_3d, valid_mask = _gen_points(extrinsic_mats)

        # Project points to image
        projected_points = project_points_to_image(
            points_3d, projection_mats
        )  # (B, P, 3)

        # Unproject points back to 3D
        unprojected_points = unproject_image_points(
            projected_points, projection_mats
        )  # (B, P, 3)

        from_image_unpoj = (
            torch.eye(4).to(device).expand(batch_size, 4, 4).clone()
        )
        from_image_unpoj[:, :3, :4] = projection_mats
        from_image_unpoj = torch.linalg.inv(from_image_unpoj)

        unprojected_points_use_inv = unproject_image_points(
            projected_points,
            to_image_proj=None,
            from_image_unpoj=from_image_unpoj,
        )

        points_3d = points_3d[valid_mask]
        unprojected_points = unprojected_points[valid_mask]
        unprojected_points_use_inv = unprojected_points_use_inv[valid_mask]

        # Check if the unprojected points are close to the original points
        assert torch.allclose(points_3d, unprojected_points, atol=1e-4), (
            "Unprojected points do not match original points"
        )
        assert torch.allclose(
            points_3d, unprojected_points_use_inv, atol=1e-4
        ), "Unprojected points do not match original points"

        # test CameraInfo unprojection
        cam_info = BatchCameraInfo(
            intrinsic_matrices=intrinsic_mats,
            pose=BatchFrameTransform(
                xyz=trans_v,
                quat=rot_q,
                parent_frame_id="camera",
                child_frame_id="world",
            ).inverse(),
            frame_id="camera",
        )
        cam_unprojected_points = cam_info.unproject_image_points(
            projected_points, frame_id="world"
        )
        cam_unprojected_points = cam_unprojected_points[valid_mask]
        assert torch.allclose(
            points_3d,
            cam_unprojected_points,
            atol=1e-3,
        ), (
            "Unprojected points do not match BatchCameraInfo"
            " unprojection results"
        )

    @pytest.mark.parametrize(
        "mat_shape, point_batch, mat_batch",
        [
            pytest.param("3x3", 5, 5),
            pytest.param("3x4", 5, 5),
            pytest.param("4x4", 5, 5),
            pytest.param("3x3", 1, 5),
            pytest.param("3x4", 1, 5),
            pytest.param("4x4", 1, 5),
            pytest.param("3x3", 5, 1),
            pytest.param("3x4", 5, 1),
            pytest.param("4x4", 5, 1),
        ],
    )
    def test_project_unproject_diff_shape(
        self,
        mat_shape: Literal["3x3", "3x4", "4x4"],
        point_batch: int,
        mat_batch: int,
    ):
        num_points = 100
        device = "cpu"
        # Random 3D points
        points_3d = (
            torch.rand(
                size=(point_batch, num_points, 3),
                device=device,
                dtype=torch.double,
            )
            - 0.5
        ) * 100.0

        # Random intrinsic matrix
        fx = (
            torch.rand(mat_batch, device=device, dtype=torch.double) * 800
            + 200
        )
        fy = (
            torch.rand(mat_batch, device=device, dtype=torch.double) * 800
            + 200
        )
        cx = torch.rand(mat_batch, device=device, dtype=torch.double) * 640
        cy = torch.rand(mat_batch, device=device, dtype=torch.double) * 480

        intrinsic_mats = torch.zeros(
            mat_batch, 3, 3, device=device, dtype=torch.double
        )
        intrinsic_mats[:, 0, 0] = fx
        intrinsic_mats[:, 1, 1] = fy
        intrinsic_mats[:, 0, 2] = cx
        intrinsic_mats[:, 1, 2] = cy
        intrinsic_mats[:, 2, 2] = 1.0

        if mat_shape != "3x3":
            trans_v = (
                torch.rand(
                    size=(mat_batch, 3), device=device, dtype=torch.double
                )
                - 0.5
            ) * 100.0

            rot_q = math_utils.normalize(
                torch.rand(
                    size=(mat_batch, 4), device=device, dtype=torch.double
                )
                - 0.5,
                dim=-1,
            )

            extrinsic_mats = transform3d.Transform3D_M.from_quat_trans(
                Q=rot_q, T=trans_v
            ).get_matrix()
            if mat_shape == "3x4":
                projection_mats = intrinsic_mats @ extrinsic_mats[:, :3, :4]
            elif mat_shape == "4x4":
                projection_mats = extrinsic_mats

        else:
            projection_mats = intrinsic_mats

        # Project points to image
        projected_points = project_points_to_image(
            points_3d, projection_mats
        )  # (B, P, 3)

        # Unproject points back to 3D
        unprojected_points = unproject_image_points(
            projected_points, projection_mats
        )  # (B, P, 3)

        # Expand points_3d if needed. This will happen when
        # broadcasting occurs in project/unproject functions
        if unprojected_points.shape[0] != points_3d.shape[0]:
            points_3d = points_3d.expand_as(unprojected_points)

        # Check if the unprojected points are close to the original points
        assert torch.allclose(points_3d, unprojected_points, atol=1e-4), (
            "Unprojected points do not match original points"
        )

    def test_get_cam_origin(self):
        device = "cpu"
        batch_size = 6

        # Random intrinsic matrix
        fx = (
            torch.rand(batch_size, device=device, dtype=torch.double) * 800
            + 200
        )
        fy = (
            torch.rand(batch_size, device=device, dtype=torch.double) * 800
            + 200
        )
        cx = torch.rand(batch_size, device=device, dtype=torch.double) * 640
        cy = torch.rand(batch_size, device=device, dtype=torch.double) * 480

        intrinsic_mats = torch.zeros(
            batch_size, 3, 3, device=device, dtype=torch.double
        )
        intrinsic_mats[:, 0, 0] = fx
        intrinsic_mats[:, 1, 1] = fy
        intrinsic_mats[:, 0, 2] = cx
        intrinsic_mats[:, 1, 2] = cy
        intrinsic_mats[:, 2, 2] = 1.0

        trans_v = (
            torch.rand(size=(batch_size, 3), device=device, dtype=torch.double)
            - 0.5
        ) * 100.0

        rot_q = math_utils.normalize(
            torch.rand(size=(batch_size, 4), device=device, dtype=torch.double)
            - 0.5,
            dim=-1,
        )
        extrinsic_mats = transform3d.Transform3D_M.from_quat_trans(
            Q=rot_q, T=trans_v
        ).get_matrix()[:, :3, :4]

        projection_mats = intrinsic_mats @ extrinsic_mats

        # Unproject points back to 3D
        unprojected_points = unproject_image_points(
            torch.tensor([[0, 0, 0]], dtype=torch.double, device=device),
            projection_mats,
        )  # (B, 1, 3)
        # unsqueeze
        unprojected_points = unprojected_points.squeeze(1)  # (B, 3)

        origin = (
            transform3d.Transform3D_M.from_quat_trans(Q=rot_q, T=trans_v)
            .inverse()
            .get_matrix()[:, :3, 3]
        )

        assert torch.allclose(
            unprojected_points,
            origin,
        ), "Camera origin does not match unprojected (0,0,0) point"
