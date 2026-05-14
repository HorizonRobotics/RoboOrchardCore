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
    BatchImageData,
    Distortion,
    ImageChannelLayout,
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

    def test_resize2d_area_matches_cv2_resize(self):
        sensor_data = torch.arange(5 * 6 * 3, dtype=torch.uint8).view(
            1, 5, 6, 3
        )
        data = BatchImageData(
            sensor_data=sensor_data,
            pix_fmt=ImageMode.RGB,
            timestamps=[11],
        )

        resized = data.resize2d(target_hw=(2, 3), inter_mode="area")

        expected = cv2.resize(
            sensor_data[0].numpy(),
            dsize=(3, 2),
            interpolation=cv2.INTER_AREA,
        )
        assert resized.sensor_data.dtype == torch.uint8
        assert resized.pix_fmt == ImageMode.RGB
        assert resized.timestamps == [11]
        assert torch.equal(
            resized.sensor_data,
            torch.asarray(expected).unsqueeze(0),
        )

    def test_resize2d_nearest_preserves_uint16_depth_values(self):
        sensor_data = torch.tensor(
            [
                [
                    [[0], [100], [200], [300]],
                    [[400], [500], [600], [700]],
                    [[800], [900], [1000], [1100]],
                    [[1200], [1300], [1400], [1500]],
                    [[1600], [1700], [1800], [1900]],
                ]
            ],
            dtype=torch.uint16,
        )
        data = BatchImageData(
            sensor_data=sensor_data,
            pix_fmt=ImageMode.I16,
        )

        resized = data.resize2d(target_hw=(2, 2), inter_mode="nearest")

        expected = cv2.resize(
            sensor_data[0, ..., 0].numpy(),
            dsize=(2, 2),
            interpolation=cv2.INTER_NEAREST,
        )
        assert resized.sensor_data.dtype == torch.uint16
        assert torch.equal(
            resized.sensor_data,
            torch.asarray(expected).unsqueeze(0).unsqueeze(-1),
        )

    def test_camera_resize2d_updates_image_shape_and_effective_intrinsic(self):
        sensor_data = torch.arange(2 * 5 * 6 * 3, dtype=torch.uint8).view(
            2, 5, 6, 3
        )
        intrinsic_matrices = torch.tensor(
            [
                [[100.0, 0.0, 3.0], [0.0, 80.0, 2.0], [0.0, 0.0, 1.0]],
                [[120.0, 0.0, 2.5], [0.0, 90.0, 1.5], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        transform_matrices = torch.tensor(
            [
                [[1.2, 0.0, 2.0], [0.0, 1.4, 3.0], [0.0, 0.0, 1.0]],
                [[0.8, 0.0, 4.0], [0.0, 1.1, 5.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        data = BatchCameraData(
            sensor_data=sensor_data,
            pix_fmt=ImageMode.RGB,
            intrinsic_matrices=intrinsic_matrices,
            transform_matrices=transform_matrices,
            frame_id="camera",
            timestamps=[101, 102],
        )

        resized = data.resize2d(target_hw=(2, 3), inter_mode="area")

        scale_mat = torch.tensor(
            [[0.5, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        expected_transform = scale_mat.unsqueeze(0) @ transform_matrices
        assert resized.image_shape == (2, 3)
        assert resized.sensor_data.shape == (2, 2, 3, 3)
        assert resized.frame_id == data.frame_id
        assert resized.timestamps == data.timestamps
        assert resized.transform_matrices is not None
        assert torch.allclose(
            resized.transform_matrices, expected_transform, atol=1e-6
        )
        resized_intrinsic = resized.get_intrinsic_with_transform()
        assert resized_intrinsic is not None
        assert torch.allclose(
            resized_intrinsic,
            expected_transform @ intrinsic_matrices,
            atol=1e-6,
        )

    def test_camera_resize2d_records_batch_scale_without_existing_transform(
        self,
    ):
        sensor_data = torch.arange(2 * 5 * 6 * 3, dtype=torch.uint8).view(
            2, 5, 6, 3
        )
        intrinsic_matrices = torch.tensor(
            [
                [[100.0, 0.0, 3.0], [0.0, 80.0, 2.0], [0.0, 0.0, 1.0]],
                [[120.0, 0.0, 2.5], [0.0, 90.0, 1.5], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        data = BatchCameraData(
            sensor_data=sensor_data,
            pix_fmt=ImageMode.RGB,
            intrinsic_matrices=intrinsic_matrices,
        )

        resized = data.resize2d(target_hw=(2, 3), inter_mode="area")

        expected_transform = (
            torch.tensor(
                [[0.5, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(2, 1, 1)
        )
        assert resized.transform_matrices is not None
        assert torch.equal(resized.transform_matrices, expected_transform)
        resized_intrinsic = resized.get_intrinsic_with_transform()
        assert resized_intrinsic is not None
        assert torch.allclose(
            resized_intrinsic,
            expected_transform @ intrinsic_matrices,
            atol=1e-6,
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
            distortion=Distortion(
                model="plumb_bob",
                coefficients=torch.tensor(
                    [[0.1, 0.01, 0.001, 0.0001]] * 2,
                    dtype=torch.float32,
                ),
            ),
        )
        aligned_data = data.to(dtype=torch.float64)

        assert aligned_data is not data
        assert data.sensor_data.dtype == torch.uint8
        assert data.intrinsic_matrices is not None
        assert data.intrinsic_matrices.dtype == torch.float32
        assert data.distortion is not None
        assert data.distortion.coefficients is not None
        assert data.distortion.coefficients.dtype == torch.float32

        assert aligned_data.sensor_data.dtype == torch.uint8
        assert aligned_data.intrinsic_matrices is not None
        assert aligned_data.intrinsic_matrices.dtype == torch.float64
        assert aligned_data.distortion is not None
        assert aligned_data.distortion.coefficients is not None
        assert aligned_data.distortion.coefficients.dtype == torch.float64

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
        aligned_data = data.to(
            dtype=torch.float64, dtype_exclude_fields=["intrinsic_matrices"]
        )

        assert aligned_data is data
        assert data.sensor_data.dtype == torch.uint8
        assert data.intrinsic_matrices is not None
        assert data.intrinsic_matrices.dtype == torch.float32

    def test_to_inplace_true_mutates_self(self):
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
            distortion=Distortion(
                model="plumb_bob",
                coefficients=torch.tensor(
                    [[0.1, 0.01, 0.001, 0.0001]] * 2,
                    dtype=torch.float32,
                ),
            ),
        )

        aligned_data = data.to(dtype=torch.float64, inplace=True)

        assert aligned_data is data
        assert data.sensor_data.dtype == torch.uint8
        assert data.intrinsic_matrices is not None
        assert data.intrinsic_matrices.dtype == torch.float64
        assert data.distortion is not None
        assert data.distortion.coefficients is not None
        assert data.distortion.coefficients.dtype == torch.float64

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

    def test_encode_decode_roundtrip_preserves_shared_fields(self):
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
            frame_id="camera",
            timestamps=[11, 22],
        )

        encoded = data.encode(format="png")
        decoded = encoded.decode()

        assert encoded.frame_id == data.frame_id
        assert encoded.timestamps == data.timestamps
        assert decoded.frame_id == data.frame_id
        assert decoded.timestamps == data.timestamps
        assert decoded.intrinsic_matrices is not None
        assert data.intrinsic_matrices is not None
        assert torch.equal(decoded.intrinsic_matrices, data.intrinsic_matrices)

    def test_encode_passes_default_codec_options(self, monkeypatch):
        from PIL import Image

        data = BatchCameraData(
            sensor_data=torch.randint(
                low=0,
                high=255,
                size=(1, 8, 7, 3),
                dtype=torch.uint8,
            ),
            pix_fmt=ImageMode.RGB,
        )
        save_calls = []
        original_save = Image.Image.save

        def save_with_capture(self, fp, format=None, **params):
            save_calls.append((format, dict(params)))
            return original_save(self, fp, format=format, **params)

        monkeypatch.setattr(Image.Image, "save", save_with_capture)

        data.encode(format="jpg", jpeg_quality=87)
        data.encode(format="png", png_compression=2)

        assert save_calls[0] == ("JPEG", {"quality": 87})
        assert save_calls[1] == ("PNG", {"compress_level": 2})

    def test_encode_rejects_default_options_for_other_formats(self):
        data = BatchCameraData(
            sensor_data=torch.randint(
                low=0,
                high=255,
                size=(1, 8, 7, 3),
                dtype=torch.uint8,
            ),
            pix_fmt=ImageMode.RGB,
        )

        with pytest.raises(ValueError, match="jpeg_quality"):
            data.encode(format="png", jpeg_quality=87)

        with pytest.raises(ValueError, match="png_compression"):
            data.encode(format="jpg", png_compression=2)

    def test_encode_rejects_default_options_with_custom_encoder(self):
        data = BatchCameraData(
            sensor_data=torch.randint(
                low=0,
                high=255,
                size=(1, 8, 7, 3),
                dtype=torch.uint8,
            ),
            pix_fmt=ImageMode.RGB,
        )

        def encoder(format: str, data: BatchImageData) -> list[bytes]:
            return [b"custom"] * data.sensor_data.shape[0]

        with pytest.raises(ValueError, match="only supported"):
            data.encode(format="png", encoder=encoder, png_compression=2)

    def test_jpg_format_is_jpeg_alias_for_encode_decode(self):
        data = BatchCameraData(
            sensor_data=torch.randint(
                low=0,
                high=255,
                size=(2, 8, 7, 3),
                dtype=torch.uint8,
            ),
            pix_fmt=ImageMode.RGB,
            intrinsic_matrices=(
                torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
            ),
            frame_id="camera",
            timestamps=[11, 22],
        )

        encoded = data.encode(format="jpg")
        decoded = encoded.decode()

        assert encoded.format == "jpg"
        assert decoded.image_shape == data.image_shape
        assert decoded.pix_fmt == ImageMode.RGB
        assert decoded.sensor_data.shape == data.sensor_data.shape
        assert decoded.frame_id == data.frame_id
        assert decoded.timestamps == data.timestamps

    def test_image_shape_hint_must_match_valid_channel_layout(self):
        with pytest.raises(ValueError, match="image shape"):
            BatchCameraData(
                sensor_data=torch.zeros((1, 5, 6, 3), dtype=torch.uint8),
                pix_fmt=ImageMode.RGB,
                image_shape=(6, 3),
            )

        with pytest.raises(ValueError, match="image shape"):
            BatchCameraData(
                sensor_data=torch.zeros((1, 3, 5, 6), dtype=torch.uint8),
                pix_fmt=ImageMode.RGB,
                image_shape=(3, 5),
            )

    def test_encoded_resize2d_uses_image_shape_for_ambiguous_hwc(self):
        data = BatchCameraData(
            sensor_data=torch.arange(1 * 3 * 6 * 3, dtype=torch.uint8).view(
                1, 3, 6, 3
            ),
            pix_fmt=ImageMode.RGB,
            image_shape=(3, 6),
        )
        encoded = data.encode(
            format="png",
            channel_layout=ImageChannelLayout.HWC,
        )

        resized = encoded.resize2d(
            target_hw=(2, 4),
            inter_mode="nearest",
        )

        decoded = resized.decode()
        assert resized.image_shape == (2, 4)
        assert decoded.sensor_data.shape == (1, 2, 4, 3)

    def test_encoded_resize2d_forwards_custom_codecs(self):
        encoded = BatchCameraDataEncoded(
            sensor_data=[b"raw"],
            format="png",
            image_shape=(3, 4),
        )
        decoded_data = BatchImageData(
            sensor_data=torch.arange(1 * 3 * 4 * 3, dtype=torch.uint8).view(
                1, 3, 4, 3
            ),
            pix_fmt=ImageMode.RGB,
        )
        encoder_calls = []

        def decoder(
            compressed_data: list[bytes],
            format: str,
        ) -> BatchImageData:
            assert compressed_data == [b"raw"]
            assert format == "png"
            return decoded_data

        def encoder(format: str, data: BatchImageData) -> list[bytes]:
            encoder_calls.append((format, tuple(data.sensor_data.shape)))
            return [b"resized"]

        resized = encoded.resize2d(
            target_hw=(2, 2),
            inter_mode="nearest",
            decoder=decoder,
            encoder=encoder,
        )

        assert resized.sensor_data == [b"resized"]
        assert resized.image_shape == (2, 2)
        assert encoder_calls == [("png", (1, 2, 2, 3))]

    def test_encoded_resize2d_preserves_format_and_updates_effective_intrinsic(
        self,
    ):
        sensor_data = torch.arange(5 * 6 * 3, dtype=torch.uint8).view(
            1, 5, 6, 3
        )
        intrinsic_matrices = torch.tensor(
            [[[100.0, 0.0, 3.0], [0.0, 80.0, 2.5], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        data = BatchCameraData(
            sensor_data=sensor_data,
            pix_fmt=ImageMode.RGB,
            intrinsic_matrices=intrinsic_matrices,
            frame_id="camera",
            timestamps=[123],
        )
        encoded = data.encode(format="png")

        resized = encoded.resize2d(
            target_hw=(3, 2),
            inter_mode="area",
            png_compression=3,
        )

        assert resized.format == "png"
        assert resized.image_shape == (3, 2)
        assert resized.frame_id == data.frame_id
        assert resized.timestamps == data.timestamps
        decoded = resized.decode()
        assert decoded.sensor_data.shape == (1, 3, 2, 3)
        expected_scale = torch.tensor(
            [
                [
                    [2.0 / 6.0, 0.0, 0.0],
                    [0.0, 3.0 / 5.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        resized_intrinsic = resized.get_intrinsic_with_transform()
        assert resized_intrinsic is not None
        assert torch.allclose(
            resized_intrinsic,
            expected_scale @ intrinsic_matrices,
            atol=1e-6,
        )

    def test_encoded_resize2d_nearest_preserves_uint16_png_depth(self):
        sensor_data = torch.tensor(
            [
                [
                    [[0], [100], [200], [300]],
                    [[400], [500], [600], [700]],
                    [[800], [900], [1000], [1100]],
                    [[1200], [1300], [1400], [1500]],
                    [[1600], [1700], [1800], [1900]],
                ]
            ],
            dtype=torch.uint16,
        )
        data = BatchCameraData(
            sensor_data=sensor_data,
            pix_fmt=ImageMode.I16,
            timestamps=[456],
        )
        encoded = data.encode(format="png", png_compression=3)

        resized = encoded.resize2d(
            target_hw=(2, 2),
            inter_mode="nearest",
            expected_sensor_dtype=torch.uint16,
            png_compression=3,
        )

        expected = cv2.resize(
            sensor_data[0, ..., 0].numpy(),
            dsize=(2, 2),
            interpolation=cv2.INTER_NEAREST,
        )
        decoded = resized.decode()
        assert resized.image_shape == (2, 2)
        assert decoded.sensor_data.dtype == torch.uint16
        assert decoded.timestamps == [456]
        assert torch.equal(
            decoded.sensor_data,
            torch.asarray(expected).unsqueeze(0).unsqueeze(-1),
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
