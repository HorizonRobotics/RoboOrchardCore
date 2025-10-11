# ruff: noqa: E501 D415 D205

"""Tutorial 2: Managing Coordinate Systems with TF Graph
====================================================================

This tutorial builds directly on the concepts from Tutorial 1. While a single
transform represents one relationship, a real robot has dozens. A
:py:class:`~robo_orchard_core.datatypes.tf_graph.BatchFrameTransformGraph`
is a powerful tool, similar to ROS TF, for
managing this entire network of coordinate frames.

You will learn to:

1.  Use :py:class:`~robo_orchard_core.datatypes.geometry.BatchFrameTransform` to define explicit parent-child relationships.

2.  Build a graph representing a simple robot arm.

3.  Query the graph to get transforms between any two frames in the system.

4.  Visualize the entire robot's coordinate frame structure.

"""

# %%
# Setup and Imports
# -----------------
# We begin by importing the necessary libraries.

import matplotlib.pyplot as plt
import numpy as np
import torch

from robo_orchard_core.datatypes.geometry import (
    BatchFrameTransform,
    BatchTransform3D,
)
from robo_orchard_core.datatypes.tf_graph import BatchFrameTransformGraph

# %%
# Defining a Robot with `BatchFrameTransform`
# ---------------------------------------------
# Unlike a generic :py:class:`~robo_orchard_core.datatypes.geometry.BatchTransform3D`,
# a :py:class:`~robo_orchard_core.datatypes.geometry.BatchFrameTransform` is semantically
# precise. It requires you to specify the `parent_frame_id` and `child_frame_id`,
# which eliminates ambiguity. A transform always describes the pose of the
# **child** frame as measured from the **parent** frame.
#
# Let's model a simple robotic arm with the following structure:
# world -> robot_base -> arm_link -> gripper

# 1. World to Robot Base: The robot's base is at (1, 0.5, 0) in the world.
world_to_base_tf = BatchFrameTransform(
    parent_frame_id="world",
    child_frame_id="robot_base",
    xyz=torch.tensor([[1.0, 0.5, 0.0]]),
    quat=torch.tensor([[1.0, 0, 0, 0]]),  # No rotation relative to world
)

# 2. Base to Arm Link: The arm begins 0.5m above the base and is rotated
#    90 degrees around the base's Z-axis.
angle_z_90 = np.pi / 2
quat_z_90 = torch.tensor(
    [[np.cos(angle_z_90 / 2), 0, 0, np.sin(angle_z_90 / 2)]]
)
base_to_arm_tf = BatchFrameTransform(
    parent_frame_id="robot_base",
    child_frame_id="arm_link",
    xyz=torch.tensor([[0.0, 0.0, 0.5]]),  # Move 0.5m up along base's Z
    quat=quat_z_90,
)

# 3. Arm Link to Gripper: The gripper is 1m away from the arm link's origin,
#    along the arm's new X-axis.
arm_to_gripper_tf = BatchFrameTransform(
    parent_frame_id="arm_link",
    child_frame_id="gripper",
    xyz=torch.tensor([[1.0, 0, 0]]),  # 1m forward in the arm's frame
    quat=torch.tensor([[1.0, 0, 0, 0]]),
)

# %%
# Building the Graph from Transformations
# ---------------------------------------
# Now that we have defined the individual relationships, we can construct the
# :py:class:`~robo_orchard_core.datatypes.tf_graph.BatchFrameTransformGraph`.
# The graph will internally build a network structure
# that understands how all these frames are connected.

# Create the graph from our list of transformations
tf_list = [world_to_base_tf, base_to_arm_tf, arm_to_gripper_tf]
tf_graph = BatchFrameTransformGraph(tf_list=tf_list)

print(
    "TF Graph created successfully with the following nodes (frames): {}".format(
        list(tf_graph.nodes.keys())
    )
)

# %%
# Querying the Graph for Any Transformation
# -----------------------------------------
# The graph's most powerful feature is its ability to answer the question:
# "What is the transformation from any frame A to any frame B?"
# It automatically finds the shortest path in the network and composes all the
# intermediate transforms.
#
# Let's find the gripper's pose directly in the world frame.

# The graph will compute this by chaining: gripper -> arm_link -> robot_base -> world
world_to_gripper_tf = tf_graph.get_tf(
    parent_frame_id="world", child_frame_id="gripper"
)

print("Gripper pose in world frame (queried from graph):")
print(f"  Position (XYZ): {world_to_gripper_tf.xyz.numpy().round(3)}")
print(f"  Orientation (Quat): {world_to_gripper_tf.quat.numpy().round(3)}")

# The graph also automatically stores and provides inverse transforms.
# Let's ask the opposite: where is the world origin as seen from the gripper?
gripper_to_world_tf = tf_graph.get_tf(
    parent_frame_id="gripper", child_frame_id="world"
)
print("World origin pose in gripper frame (inverse query):")
print(f"  Position (XYZ): {gripper_to_world_tf.xyz.numpy().round(3)}")

# %%
# Visualizing the Entire Frame Network
# ------------------------------------
# A 3D plot makes the spatial relationships between all the frames instantly clear.
# We can iterate through every frame in our graph and plot its pose relative to
# the 'world' frame.

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")


def plot_frame(ax, transform, name, length=0.4):
    """Helper function to plot a 3D coordinate frame."""
    # Define the origin and the endpoints of the three axes in the local frame
    origin_local = torch.tensor([[[0.0, 0.0, 0.0]]])
    x_axis_local = torch.tensor([[[length, 0.0, 0.0]]])
    y_axis_local = torch.tensor([[[0.0, length, 0.0]]])
    z_axis_local = torch.tensor([[[0.0, 0.0, length]]])

    # Transform these points into the world frame and extract the single coordinate vector
    # The first [0] selects the batch, the second [0] extracts the point from the batch.
    # This changes the shape from (1, 3) to (3,).
    origin = transform.transform_points(origin_local)[0][0]
    x_axis_end = transform.transform_points(x_axis_local)[0][0]
    y_axis_end = transform.transform_points(y_axis_local)[0][0]
    z_axis_end = transform.transform_points(z_axis_local)[0][0]

    # Plot axes: X=Red, Y=Green, Z=Blue
    # Now, origin[0], origin[1], and origin[2] correctly access the x, y, z values.
    ax.plot(
        [origin[0], x_axis_end[0]],
        [origin[1], x_axis_end[1]],
        [origin[2], x_axis_end[2]],
        color="r",
    )
    ax.plot(
        [origin[0], y_axis_end[0]],
        [origin[1], y_axis_end[1]],
        [origin[2], y_axis_end[2]],
        color="g",
    )
    ax.plot(
        [origin[0], z_axis_end[0]],
        [origin[1], z_axis_end[1]],
        [origin[2], z_axis_end[2]],
        color="b",
    )
    ax.text(origin[0], origin[1], origin[2], f"  {name}", fontsize=12)


# Plot the world frame at the origin using an identity transform
plot_frame(ax, BatchTransform3D.identity(1), "world")

# Plot every other frame by querying its pose in the world
for frame_id in tf_graph.nodes:
    if frame_id != "world":
        # Query the transform from 'world' to the current frame
        world_to_frame_tf = tf_graph.get_tf("world", frame_id)
        plot_frame(ax, world_to_frame_tf, frame_id)

ax.set_xlabel("World X-axis")
ax.set_ylabel("World Y-axis")
ax.set_zlabel("World Z-axis")
ax.set_title("Coordinate Frames of the Robot System")

# Set aspect ratio to be equal
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-1, 2.5])
ax.set_ylim([-1, 2])
ax.set_zlim([0, 2])
ax.view_init(elev=20.0, azim=-60)  # Set a nice viewing angle
plt.grid(True)
plt.show()
