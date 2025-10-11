# ruff: noqa: E501 D415 D205

"""Tutorial 3: Representing Robot Joints
=====================================================

This tutorial covers :py:class:`~robo_orchard_core.datatypes.joint_state.BatchJointsState`, the standard dataclass for representing
the state of a robot's articulated joints in `robo_orchard_core`. This object is
a container for time-series data like joint positions, velocities, and efforts.

You will learn to:

1.  Create a :py:class:`~robo_orchard_core.datatypes.joint_state.BatchJointsState` object for a trajectory of joint states.

2.  Understand its core properties (`position`, `velocity`, `effort`, `names`).

3.  Visualize joint trajectories using Matplotlib.

4.  Perform powerful slicing and indexing operations to extract data.

5.  Concatenate multiple joint state objects to combine trajectories.
"""

# %%
# Setup and Imports
# -----------------
# We'll need `torch` to create our data and `matplotlib` for visualization.

import matplotlib.pyplot as plt
import numpy as np
import torch

from robo_orchard_core.datatypes.joint_state import BatchJointsState

# %%
# Creating a Batch of Joint States
# --------------------------------
# A :py:class:`~robo_orchard_core.datatypes.joint_state.BatchJointsState`
# is designed to hold a "batch" of states. In robotics,
# this batch often represents a time series or a trajectory.
#
# Let's model a 6-joint robot arm over a trajectory of 100 timesteps. The
# resulting object will have a `batch_size` of 100 and a `joint_num` of 6.

batch_size = 100  # Number of timesteps in the trajectory
num_joints = 6  # Number of joints in the robot

# We will generate a synthetic trajectory using sinusoidal functions to simulate
# smooth joint movements.
time = torch.linspace(0, 4 * np.pi, batch_size).unsqueeze(1)
# Each joint will have a slightly phase-shifted sine wave for its position.
positions = torch.sin(time + torch.linspace(0, np.pi, num_joints))
# The velocity is the derivative of the position (cosine).
velocities = torch.cos(time + torch.linspace(0, np.pi, num_joints))

# It's crucial to name the joints. The order of names must match the columns
# in the position/velocity/effort tensors.
joint_names = [f"joint_{i + 1}" for i in range(num_joints)]

# Now, we instantiate the object. Note that `effort` is optional, so we can
# leave it as None if we don't have that data.
trajectory = BatchJointsState(
    position=positions, velocity=velocities, names=joint_names
)

print(
    f"Created a trajectory with batch_size={trajectory.batch_size}, num_joints={trajectory.joint_num} and Joint names: {trajectory.names}"
)


# %%
# Visualizing Joint Trajectories
# ------------------------------
# A great way to understand the data is to plot it. Let's visualize the
# position of the first three joints over the entire trajectory. The batch
# dimension (time) will be our x-axis.

plt.figure(figsize=(10, 6))
plt.title("Position of First 3 Joints Over Time")
plt.xlabel("Timestep (Batch Index)")
plt.ylabel("Position (radians)")

# We can easily access the position data and plot each joint's column
for i in range(3):
    plt.plot(trajectory.position[:, i], label=trajectory.names[i])

plt.legend()
plt.grid(True)
plt.show()

# %%
# Slicing and Indexing
# --------------------
# The :py:class:`~robo_orchard_core.datatypes.joint_state.BatchJointsState`
# object behaves like a smart tensor, allowing for
# intuitive slicing and indexing. The returned object is always a valid
# :py:class:`~robo_orchard_core.datatypes.joint_state.BatchJointsState`.

# --- Example 1: Get the state at a single timestep ---
# This returns a new BatchJointsState with a batch_size of 1.
state_at_t10 = trajectory[10]
print(f"State at timestep 10 (type: {type(state_at_t10)}):")
print(f"  Batch size: {state_at_t10.batch_size}")
print(f"  Position: {state_at_t10.position.numpy().round(2)}")

# --- Example 2: Get a slice of the trajectory for only the wrist joints ---
# Let's say the last 3 joints form the wrist. We can slice both dimensions.
# : in the first slot means "all timesteps".
# 3: in the second slot means "from the 4th joint to the end".
wrist_trajectory = trajectory[:, 3:]
print("Created a new state object for only the wrist joints:")
print(f"  Batch size: {wrist_trajectory.batch_size}")
print(f"  Number of joints: {wrist_trajectory.joint_num}")
print(f"  Joint names: {wrist_trajectory.names}")

# %%
# Concatenating Joint States
# --------------------------
# Often, you may need to combine different trajectories. The `concat` class
# method makes this straightforward.
#
# Let's create a second, shorter trajectory and append it to our first one.

# A new trajectory of 50 timesteps
new_time = torch.linspace(4 * np.pi, 6 * np.pi, 50).unsqueeze(1)
new_positions = torch.sin(new_time + torch.linspace(0, np.pi, num_joints))
new_velocities = torch.cos(new_time + torch.linspace(0, np.pi, num_joints))

trajectory_part2 = BatchJointsState(
    position=new_positions,
    velocity=new_velocities,
    names=joint_names,  # Names must match to concatenate along the batch dim
)

# Concatenate along the batch dimension (dim=0)
combined_trajectory = BatchJointsState.concat(
    [trajectory, trajectory_part2], dim=0
)

print(f"Original trajectory length: {trajectory.batch_size}")
print(f"Second trajectory length: {trajectory_part2.batch_size}")
print(f"Combined trajectory length: {combined_trajectory.batch_size}")

# Let's plot joint_1's position from the combined trajectory to see the result.
plt.figure(figsize=(10, 6))
plt.title("Combined Trajectory for joint_1")
plt.xlabel("Timestep (Batch Index)")
plt.ylabel("Position (radians)")
plt.plot(
    combined_trajectory.position[:, 0], label=combined_trajectory.names[0]
)
# Add a vertical line to show where the concatenation happened
plt.axvline(
    x=trajectory.batch_size,
    color="r",
    linestyle="--",
    label="Concatenation Point",
)
plt.legend()
plt.grid(True)
plt.show()
