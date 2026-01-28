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
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from robo_orchard_core.utils.math import math_utils, transform3d


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # check input is correct
    if q1.shape != q2.shape:
        msg = (
            f"Expected input quaternion shape mismatch: {q1.shape} "
            + f"!= {q2.shape}."
        )
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def quat_from_angle_axis(
    angle: torch.Tensor, axis: torch.Tensor
) -> torch.Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


def apply_delta_pose(
    source_pos: torch.Tensor,
    source_rot: torch.Tensor,
    delta_pose: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isaaclab apply delta pose impl."""
    # number of poses given
    num_poses = source_pos.shape[0]
    device = source_pos.device

    # interpret delta_pose[:, 0:3] as target position displacements
    target_pos = source_pos + delta_pose[:, 0:3]
    # interpret delta_pose[:, 3:6] as target rotation displacements
    rot_actions = delta_pose[:, 3:6]
    angle = torch.linalg.vector_norm(rot_actions, dim=1)
    axis = rot_actions / angle.unsqueeze(-1)
    # change from axis-angle to quat convention
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(
        num_poses, 1
    )
    rot_delta_quat = torch.where(
        angle.unsqueeze(-1).repeat(1, 4) > eps,
        quat_from_angle_axis(angle, axis),
        identity_quat,
    )
    # TODO: Check if this is the correct order for this multiplication.
    target_rot = quat_mul(rot_delta_quat, source_rot)

    return target_pos, target_rot


class TestQuaternion:
    def test_quaternion_consistency_scipy(self):
        q = math_utils.normalize(torch.rand(size=(1000, 4)) - 0.5, dim=-1)
        q = math_utils.quaternion_standardize(q)
        M = math_utils.quaternion_to_matrix(q)
        q_recovered = math_utils.matrix_to_quaternion(M)
        q_recovered = math_utils.quaternion_standardize(q_recovered)
        axis_angle = math_utils.quaternion_to_axis_angle(q)

        q_scipy: np.ndarray = Rotation.from_quat(
            q.cpu().numpy(), scalar_first=True
        ).as_matrix()
        q_scipy = q_scipy.astype(np.float32)

        q_recovered_scipy = Rotation.from_matrix(M.cpu().numpy()).as_quat(
            scalar_first=True
        )
        q_recovered_scipy = math_utils.quaternion_standardize(
            torch.from_numpy(q_recovered_scipy)
        ).to(torch.float32)

        axis_angle_scipy = Rotation.from_quat(
            q.cpu().numpy(), scalar_first=True
        ).as_rotvec()
        axis_angle_scipy = torch.from_numpy(axis_angle_scipy).to(torch.float32)

        assert torch.allclose(q, q_recovered, atol=1e-5)
        assert torch.allclose(torch.from_numpy(q_scipy), M.cpu(), atol=1e-5)
        assert torch.allclose(q_recovered_scipy, q_recovered, atol=1e-5)
        assert torch.allclose(axis_angle, axis_angle_scipy, atol=1e-5)

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
    def test_quaternion_to_rotation_matrix_consistency(self, device: str):
        q = math_utils.normalize(
            torch.rand(size=(1000, 4), device=device) - 0.5, dim=-1
        )
        q = math_utils.quaternion_standardize(q)
        M = math_utils.quaternion_to_matrix(q)
        q_recovered = math_utils.matrix_to_quaternion(M)
        q_recovered = math_utils.quaternion_standardize(q_recovered)
        assert torch.allclose(q, q_recovered, atol=1e-5)

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
    def test_quaternion_to_axis_angle_consistency(self, device: str):
        q = math_utils.normalize(
            torch.rand(size=(1000, 4), device=device) - 0.5, dim=-1
        )
        q = math_utils.quaternion_standardize(q)
        angle_axis = math_utils.quaternion_to_axis_angle(q)
        q_recovered = math_utils.axis_angle_to_quaternion(angle_axis)
        q_recovered = math_utils.quaternion_standardize(q_recovered)
        assert torch.allclose(q, q_recovered, atol=1e-5)

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
    def test_point_apply_broadcase_same_dim(self, device: str):
        batch_size = 1000
        q = math_utils.normalize(
            torch.rand(size=(1, 4), device=device) - 0.5, dim=-1
        )
        T = torch.rand(size=(1, 3), device=device) - 0.5
        p = torch.rand(size=(batch_size, 3), device=device) - 0.5
        p_rotated_ = math_utils.quaternion_apply_point(q, p) + T

        assert p_rotated_.shape == (batch_size, 3)

        # expand q and T to batch_size
        q_b = q.expand(batch_size, 4)
        T_b = T.expand(batch_size, 3)
        p_rotated = math_utils.quaternion_apply_point(q_b, p) + T_b
        assert torch.allclose(p_rotated, p_rotated_, atol=1e-6)

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
    def test_point_apply_broadcase_diff_dim(self, device: str):
        batch_size = 1000
        q = math_utils.normalize(
            torch.rand(size=(4,), device=device) - 0.5, dim=-1
        )
        T = torch.rand(size=(3,), device=device) - 0.5
        p = torch.rand(size=(batch_size, 3), device=device) - 0.5
        p_rotated_ = math_utils.quaternion_apply_point(q, p) + T

        assert p_rotated_.shape == (batch_size, 3)

        # expand q and T to batch_size
        q_b = q.expand(batch_size, 4)
        T_b = T.expand(batch_size, 3)
        p_rotated = math_utils.quaternion_apply_point(q_b, p) + T_b
        assert torch.allclose(p_rotated, p_rotated_, atol=1e-6)

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
    def test_point_apply_batch_mode_broadcast_diff_dim(self, device: str):
        batch_size = 3
        q = math_utils.normalize(
            torch.rand(size=(batch_size, 4), device=device) - 0.5, dim=-1
        )
        T = torch.rand(size=(batch_size, 1000, 3), device=device) - 0.5
        p = torch.rand(size=(batch_size, 1000, 3), device=device) - 0.5
        p_rotated = (
            math_utils.quaternion_apply_point(q, p, batch_mode=True) + T
        )
        assert p_rotated.shape == p.shape

        for i in range(batch_size):
            p_rotated_ = math_utils.quaternion_apply_point(q[i], p[i]) + T[i]
            assert torch.allclose(p_rotated[i], p_rotated_, atol=1e-6)

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
    def test_point_apply_batch_mode_broadcast_same_dim(self, device: str):
        batch_size = 3
        q = math_utils.normalize(
            torch.rand(size=(batch_size, 1, 4), device=device) - 0.5, dim=-1
        )
        T = torch.rand(size=(batch_size, 1000, 3), device=device) - 0.5
        p = torch.rand(size=(batch_size, 1000, 3), device=device) - 0.5
        p_rotated = (
            math_utils.quaternion_apply_point(q, p, batch_mode=True) + T
        )
        assert p_rotated.shape == p.shape

        for i in range(batch_size):
            p_rotated_ = math_utils.quaternion_apply_point(q[i], p[i]) + T[i]
            assert torch.allclose(p_rotated[i], p_rotated_, atol=1e-6)


class TestFrameTransform:
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
    def test_frame_transform_combine_substrace_consistency(self, device):
        q = math_utils.normalize(
            torch.rand(size=(2, 1, 4), device=device) - 0.5, dim=-1
        )
        q_02, q_01 = math_utils.quaternion_standardize(q).unbind(0)
        t = torch.rand(size=(2, 1, 3), device=device) - 0.5
        t_02, t_01 = t.unbind(0)

        t_12, q_12 = math_utils.frame_transform_subtract(
            t_01, q_01, t_02, q_02
        )

        t_02_, q_02_ = math_utils.frame_transform_combine(
            t_01, q_01, t_12, q_12
        )

        assert torch.allclose(q_02, q_02_, atol=1e-6)
        assert torch.allclose(t_02, t_02_, atol=1e-6)


class TestRotationMatrix:
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
    def test_matrix_to_axis_angle_consistency(self, device: str):
        q = math_utils.normalize(
            torch.rand(size=(1000, 4), device=device) - 0.5, dim=-1
        )
        M = math_utils.quaternion_to_matrix(q)
        angle_axis = math_utils.matrix_to_axis_angle(M)
        M_recovered = math_utils.axis_angle_to_matrix(angle_axis)
        assert torch.allclose(M, M_recovered, atol=1e-5)


class TestTransform3d:
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
    def test_compose_consistency(self, device: str):
        def random_t3d(batch_size: int):
            q = math_utils.normalize(
                torch.rand(size=(batch_size, 4), device=device) - 0.5, dim=-1
            )
            # 1000 random rotation matrices
            R = math_utils.quaternion_to_matrix(q)
            # 1000 random translations
            t = torch.rand(size=(batch_size, 3), device=device) - 0.5

            t3d = transform3d.Transform3D_M.from_rot_trans(R, t)
            return t3d

        batch_size = 1000
        t3d_1 = random_t3d(batch_size)
        t3d_2 = random_t3d(batch_size)
        t3d_3 = random_t3d(batch_size)

        p = torch.rand(size=(batch_size, 3), device=device) - 0.5

        p_t1 = t3d_3.transform_points(
            t3d_2.transform_points(t3d_1.transform_points(p))
        )
        p_t2 = t3d_1.compose(t3d_2, t3d_3).transform_points(p)
        p_t3 = t3d_1.compose(t3d_2).compose(t3d_3).transform_points(p)
        p_t3_ = (t3d_3 @ t3d_2 @ t3d_1).transform_points(p)

        assert torch.allclose(p_t1, p_t2, atol=1e-6, rtol=1e-5)
        assert torch.allclose(p_t1, p_t3, atol=1e-6, rtol=1e-5)
        assert torch.allclose(p_t1, p_t3_, atol=1e-6, rtol=1e-5)

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
    def test_point_apply_consistency(self, device: str):
        batch_size = 1000
        q = math_utils.normalize(
            torch.rand(size=(batch_size, 4), device=device) - 0.5, dim=-1
        )
        T = torch.rand(size=(batch_size, 3), device=device) - 0.5
        M = math_utils.quaternion_to_matrix(q)
        p = torch.rand(size=(batch_size, 3), device=device) - 0.5
        t3d = transform3d.Transform3D_M.from_rot_trans(M, T)

        p_rotated = t3d.transform_points(p.reshape(batch_size, 1, 3)).reshape(
            batch_size, 3
        )
        p_rotated_ = math_utils.quaternion_apply_point(q, p) + T
        assert p_rotated.shape == p_rotated_.shape
        assert torch.allclose(p_rotated, p_rotated_, atol=1e-6)

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
    def test_frame_transform_combine(self, device: str):
        q = math_utils.normalize(
            torch.rand(size=(2, 4), device=device) - 0.5, dim=-1
        )
        t = torch.rand(size=(2, 3), device=device) - 0.5

        c_t, c_q = math_utils.frame_transform_combine(
            q01=q[0], t01=t[0], q12=q[1], t12=t[1]
        )
        c_q = math_utils.quaternion_standardize(c_q)
        # equal to compose of M_1 and M_0
        # M_1 is the tf of frame 2 w.r.t frame 1, and
        # M_0 is the tf of frame 1 w.r.t frame 0
        # The combined tf is the tf of frame 2 w.r.t frame 0
        # so the combined tf is matmul(M_1, M_0)
        c_M = transform3d.Transform3D_M(device=device).compose(
            transform3d.Transform3D_M.from_rot_trans(
                R=math_utils.quaternion_to_matrix(q[1]), T=t[1]
            ),
            transform3d.Transform3D_M.from_rot_trans(
                R=math_utils.quaternion_to_matrix(q[0]), T=t[0]
            ),
        )
        c_q_, c_t_ = c_M.get_rotation_quaternion(), c_M.get_translation()
        c_q_ = math_utils.quaternion_standardize(c_q_)
        assert torch.allclose(c_q, c_q_, atol=1e-6)
        assert torch.allclose(c_t, c_t_, atol=1e-6)

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
    def test_frame_transform_subtract(self, device: str):
        q_01, q_02 = math_utils.quaternion_standardize(
            math_utils.normalize(
                torch.rand(size=(2, 4), device=device) - 0.5, dim=-1
            )
        ).unbind(0)
        t_01, t_02 = (torch.rand(size=(2, 3), device=device) - 0.5).unbind(0)

        t_12, q_12 = math_utils.frame_transform_subtract(
            t01=t_01, q01=q_01, t02=t_02, q02=q_02
        )
        # should be equal to M_01.inverse() * M_02
        c_M = transform3d.Transform3D_M(device=device).compose(
            transform3d.Transform3D_M.from_rot_trans(
                R=math_utils.quaternion_to_matrix(q_02), T=t_02
            ),
            transform3d.Transform3D_M.from_rot_trans(
                R=math_utils.quaternion_to_matrix(q_01), T=t_01
            ).inverse(),
        )
        q_12_ = math_utils.quaternion_standardize(
            c_M.get_rotation_quaternion()
        )[0]
        t_12_ = c_M.get_translation()[0]
        assert torch.allclose(q_12, q_12_, atol=1e-6)
        assert torch.allclose(t_12, t_12_, atol=1e-6)

        # check if reconstructing the original tf
        t_02_, q_02_ = math_utils.frame_transform_combine(
            t01=t_01, q01=q_01, t12=t_12, q12=q_12
        )
        q_02_ = math_utils.quaternion_standardize(q_02_)
        assert torch.allclose(q_02, q_02_, atol=1e-6)
        assert torch.allclose(t_02, t_02_, atol=1e-6)

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
    def test_pose_diff(self, device: str):
        q = math_utils.normalize(
            torch.rand(size=(2, 4), device=device) - 0.5, dim=-1
        )
        q = math_utils.quaternion_standardize(q)
        t = torch.rand(size=(2, 3), device=device) - 0.5

        diff_t, diff_q = math_utils.pose_diff(
            ta=t[0], qa=q[0], tb=t[1], qb=q[1]
        )

        assert torch.allclose(t[0], t[1] + diff_t, atol=1e-6)

        # rotation check
        assert torch.allclose(
            q[0], math_utils.quaternion_multiply(diff_q, q[1]), atol=1e-6
        )

        # check consistency with isaaclab math
        delta_pose = torch.concat(
            [diff_t, math_utils.quaternion_to_axis_angle(diff_q)], dim=-1
        ).unsqueeze(0)

        r_p, r_q = apply_delta_pose(
            source_pos=t[1].unsqueeze(0),
            source_rot=q[1].unsqueeze(0),
            delta_pose=delta_pose,
        )
        r_q = math_utils.quaternion_standardize(r_q)[0]
        r_p = r_p[0]

        assert torch.allclose(r_p, t[0], atol=1e-6)
        assert torch.allclose(r_q, q[0], atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
