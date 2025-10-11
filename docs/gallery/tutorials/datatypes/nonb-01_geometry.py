# ruff: noqa: E501 D415 D205

"""Tutorial 1: Geometry Basics with BatchTransform3D and BatchPose
=======================================================================================

This tutorial covers the most fundamental building block for all robotics tasks:
representing 3D position and orientation.

You will learn how to:

1.  Create a single 3D transformation using :py:class:`~robo_orchard_core.datatypes.geometry.BatchTransform3D`.

2.  Perform core operations like composition and inversion.

3.  Work with batches of transforms for efficient computation.

4.  Apply transforms to 3D points.

5.  Understand the semantic difference in :py:class:`~robo_orchard_core.datatypes.geometry.BatchPose`.

"""

# %%
# Setup and Imports
# -----------------
# First, let's import the necessary libraries. We'll need `torch` for tensor
# operations, `numpy` for some math constants, and `matplotlib` for visualization.

import matplotlib.pyplot as plt
import numpy as np
import torch

from robo_orchard_core.datatypes.geometry import BatchPose, BatchTransform3D

# %%
# Creating a Single 3D Transform
# ------------------------------
# A :py:class:`~robo_orchard_core.datatypes.geometry.BatchTransform3D` represents a 3D rigid body transformation.
# It consists of two key components:
#
# * A translation vector (`xyz`).
#
# * A rotation quaternion (`quat`) in (w, x, y, z) format.
#
# Let's create a transform that moves an object 2 units along the X-axis, 1 unit
# along the Y-axis, 0.5 units along the Z-axis, and then rotates it 45 degrees
# around the Z-axis.

# A 45-degree angle is pi/4 radians.
# The quaternion for a Z-axis rotation by angle 'a' is [cos(a/2), 0, 0, sin(a/2)].
angle = np.pi / 4
quat_z_45 = torch.tensor(
    [np.cos(angle / 2), 0, 0, np.sin(angle / 2)], dtype=torch.float32
)
xyz_translation = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32)

# Even for a single transform, the datatypes are "batched". So, we need to add a
# batch dimension of 1 to our tensors using `unsqueeze(0)`.
transform_a = BatchTransform3D(
    xyz=xyz_translation.unsqueeze(0), quat=quat_z_45.unsqueeze(0)
)
# A 45-degree angle is pi/4 radians.
# The quaternion for a Z-axis rotation by angle 'a' is [cos(a/2), 0, 0, sin(a/2)].
angle = np.pi / 4
quat = torch.tensor(
    [np.cos(angle / 2), 0, 0, np.sin(angle / 2)], dtype=torch.float32
)
xyz = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32)

# Even for a single transform, the datatypes are "batched". So, we need to add a
# batch dimension of 1 to our tensors using `unsqueeze(0)`.
transform_a = BatchTransform3D(xyz=xyz.unsqueeze(0), quat=quat.unsqueeze(0))

print(f"Our first transform (Transform A): {transform_a}")
print(f"Batch size: {transform_a.batch_size}")

# %%
# Core Operations: Inverse and Compose
# ------------------------------------
# Two fundamental operations on transforms are inversion and composition.
#
# * **Inverse**: Calculates the opposite transform that would move the object back
#     to its origin.
#
# * **Compose**: Chains two transforms together to create a single equivalent
#     transform. If C = A.compose(B), applying C is the same as applying A then B.

# Calculate the inverse of Transform A
transform_a_inv = transform_a.inverse()
print(f"Inverse of Transform A: {transform_a_inv}")

# Now, let's define a second transform (B) that represents a simple translation
# of -1.5 units along the Y-axis.
transform_b = BatchTransform3D(
    xyz=torch.tensor([[0.0, -1.5, 0.0]]),
    quat=torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]]
    ),  # No rotation (identity quaternion)
)

# Compose A with B to get C
transform_c = transform_a.compose(transform_b)
print(f"Transform C (A composed with B): {transform_c}")

# %%
# Working with Batches and Transforming Points
# ---------------------------------------------
# The "Batch" in :py:class:`~robo_orchard_core.datatypes.geometry.BatchTransform3D` is its key feature.
# It allows you to perform operations on many transforms simultaneously, which is highly efficient on GPUs.
#
# Let's create a batch of 4 transforms and use them to transform 4 different
# point clouds.

batch_size = 4
# Create a batch of 4 identity transforms (no translation, no rotation)
transforms_batch = BatchTransform3D.identity(batch_size, device="cpu")

# Now, let's modify some of the transforms in the batch
# Transform 0: Translate by +3 along the X-axis
transforms_batch.xyz[0, 0] = 3.0
# Transform 1: Translate by +3 along the Y-axis
transforms_batch.xyz[1, 1] = 3.0
# Transform 2: Rotate 90 degrees around X-axis
angle_90 = np.pi / 2
transforms_batch.quat[2] = torch.tensor(
    [np.cos(angle_90 / 2), np.sin(angle_90 / 2), 0, 0]
)

# Create a batch of 4 random point clouds, each with 20 points
points = torch.randn(batch_size, 20, 3)

# Apply all 4 transforms to their corresponding point clouds in one go
transformed_points = transforms_batch.transform_points(points)
print(f"Shape of original points batch: {points.shape}")
print(f"Shape of transformed points batch: {transformed_points.shape}")

# Visualize the effect of the first transform on the first point cloud
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
p_orig = points[0].numpy()
p_trans = transformed_points[0].numpy()
ax.scatter(
    p_orig[:, 0],
    p_orig[:, 1],
    p_orig[:, 2],
    label="Original Points (Batch 0)",
    c="blue",
)
ax.scatter(
    p_trans[:, 0],
    p_trans[:, 1],
    p_trans[:, 2],
    label="Transformed Points (Batch 0)",
    c="red",
    marker="^",
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.set_title("Visualizing a Batched Transformation")
ax.grid(True)
plt.show()

# %%
# Understanding BatchPose with Frame IDs
# ---------------------------------------------
# A :py:class:`~robo_orchard_core.datatypes.geometry.BatchPose` is a subclass of :py:class:`~robo_orchard_core.datatypes.geometry.BatchTransform3D`
# with an important semantic addition: a **`frame_id`**.
# This string specifies the coordinate frame in which
# the pose is defined. For example, `frame_id="world"` means the position and
# orientation are relative to the world origin.
#
# This distinction is crucial for preventing errors in complex systems. It adds
# essential context to a raw transformation.

# Let's create a pose for a robot base, specifying it is relative to the "world".
robot_base_position = torch.tensor([[0.5, -1.0, 0.0]])
# No rotation relative to the world
robot_base_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

robot_pose_in_world = BatchPose(
    xyz=robot_base_position, quat=robot_base_orientation, frame_id="world"
)

print(f"A BatchPose object with a specified frame_id: {robot_pose_in_world}")

# %%
# This simple addition of context is the key difference and the first step toward
# building robust transform trees, which we will explore in the next tutorial using
# the :py:class:`~robo_orchard_core.datatypes.geometry.BatchFrameTransform` class.
