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

from robo_orchard_core.utils.math.transform.transform3d import (
    transform_points_3d,
)


def project_points_to_image(
    points_3d: torch.Tensor,
    projection_mat: torch.Tensor,
) -> torch.Tensor:
    """Project 3D points to 2D image plane using the camera intrinsic matrix.

    Args:
        points_3d (torch.Tensor): A tensor of shape (P, 3) or (B, P, 3)
            representing the 3D points in any coordinate frame.
        projection_mat (torch.Tensor): A tensor of shape (3, 3) or (B, 3, 3),
            or (3, 4) or (B, 3, 4) , or (4, 4) or (B, 4, 4) representing
            projection matrix.

    Returns:
        torch.Tensor: A tensor of shape (B, P, 3) representing the projected
            2D points with depth (u, v, depth) in the image plane. The depth
            is orthogonal distance from the camera's image plane to the
            3D point.
    """
    if points_3d.dim() == 2:
        points_3d = points_3d[None]  # (P, 3) -> (1, P, 3)
    if projection_mat.dim() == 2:
        projection_mat = projection_mat[
            None
        ]  # (3, 3) or (3, 4) or (4, 4) -> (1, 3, 3) or (1, 3, 4) or (1, 4, 4)

    if projection_mat.shape[-1] != 4:
        # pad projection_mat to 4x4
        padded_proj_mat = (
            torch.eye(4)
            .to(projection_mat)
            .expand(projection_mat.shape[0], 4, 4)
            .clone()
        )
        padded_proj_mat[
            ..., : projection_mat.shape[-2], : projection_mat.shape[-1]
        ] = projection_mat
        projection_mat = padded_proj_mat

    # Project points using the intrinsic matrix
    projected_points = transform_points_3d(
        points_3d, projection_mat
    )  # (B, P, 3)

    # Normalize to get pixel coordinates
    projected_points[..., :2] /= projected_points[..., 2:3]

    return projected_points


def unproject_image_points(
    uvd: torch.Tensor,
    to_image_proj: torch.Tensor | None = None,
    from_image_unpoj: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unproject 2D image points with depth to 3D points in camera frame.

    Args:
        uvd (torch.Tensor): A tensor of shape (P, 3) or (B, P, 3)
            representing the 2D image points with depth (u, v, depth).
        to_image_proj (torch.Tensor | None): A tensor of shape (3, 3)
            or (B, 3, 3), or (3, 4) or (B, 3, 4) , or (4, 4) or (B, 4, 4)
            representing projection matrix to image plane. If
            projection_mat has shape (4, 4) or (B, 4, 4), the last row should
            be [0, 0, 0, 1], as the denominator of the 4D homogeneous
            coordinate is unknown.
        from_image_unpoj (torch.Tensor | None): The inverse of the projection
            matrix from camera frame to image plane. If provided, it will be
            used instead of computing the inverse of to_image_proj.

    Returns:
        torch.Tensor: A tensor of shape (B, P, 3) representing the unprojected
            3D points in camera frame.
    """
    if uvd.dim() == 2:
        uvd = uvd[None]  # (P, 3) -> (1, P, 3)

    if to_image_proj is None and from_image_unpoj is None:
        raise ValueError(
            "Either to_image_proj or from_image_unpoj must be provided."
        )
    if to_image_proj is not None:
        proj_mat = to_image_proj
        need_inverse = True
    else:
        proj_mat = from_image_unpoj
        need_inverse = False
    assert proj_mat is not None

    if proj_mat.dim() == 2:
        proj_mat = proj_mat[
            None
        ]  # (3, 3) or (3, 4) or (4, 4) -> (1, 3, 3) or (1, 3, 4) or (1, 4, 4)

    # Convert uvd to scaled pixel coordinates
    # from (u, v), (d) to (u*d, v*d, d)
    uvd = torch.concat(
        [uvd[..., :2] * uvd[..., 2:3], uvd[..., 2:3]], dim=-1
    )  # (B, P, 3)

    s = proj_mat.shape
    if s[-2] != 4 or s[-1] != 4:
        # need to convert to 4x4 matrix to solve
        padded_proj_mat = (
            torch.eye(4).to(proj_mat).expand(proj_mat.shape[0], 4, 4).clone()
        )
        padded_proj_mat[..., : s[-2], : s[-1]] = proj_mat
        proj_mat = padded_proj_mat

    if need_inverse:
        # expand to homogeneous coordinates
        ones = torch.ones(
            uvd.shape[0], uvd.shape[1], 1, dtype=uvd.dtype, device=uvd.device
        )
        uvd = torch.cat([uvd, ones], dim=-1)  # (B, P, 4)
        # the projection: points_3d_homogeneous * mat.T = uvd_homogeneous
        # use linalg.solve to solve the equation
        to_image_proj_transpose = proj_mat.transpose(-1, -2)
        points_3d_homogeneous = torch.linalg.solve(
            to_image_proj_transpose, uvd, left=False
        )
        points_3d = points_3d_homogeneous[..., :3]
    else:
        points_3d = transform_points_3d(uvd, proj_mat)
    return points_3d
