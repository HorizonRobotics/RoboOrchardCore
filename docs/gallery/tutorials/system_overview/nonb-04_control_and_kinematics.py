# ruff: noqa: E501 D415 D205

# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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

"""Controllers, Kinematics, and Motion Commands
===========================================================
"""

# %%
# Why This Is A Separate Tutorial
# -------------------------------
#
# ``robo_orchard_core`` separates robot structure, motion solving, and
# environment execution into different module areas. This separation matters
# because those concerns change for different reasons:
#
# - the robot description changes when the physical mechanism changes
# - the controller changes when the command strategy changes
# - the environment changes when the simulator or device backend changes
#
# Keeping these boundaries explicit makes the control stack easier to test,
# swap, and reason about.


# %%
# When To Read This Tutorial
# --------------------------
#
# Read this tutorial when your next question is no longer how an environment
# steps, but how a high-level motion target becomes a command that a robot or
# simulator can actually consume.


# %%
# Kinematics Describes Robot Structure
# ------------------------------------
#
# The ``kinematics`` package is responsible for robot structure and frame
# relationships. ``KinematicChain`` loads robot descriptions such as URDF,
# SDF, or MJCF and exposes a chain model that can answer pose questions from
# joint values.
#
# This layer is about *geometry and structure*, not command generation.
# Typical responsibilities include:
#
# - loading a chain description from robot model content
# - computing link or frame transforms from joint states
# - preserving frame naming and parent/child relationships
# - converting robot structure into reusable transform queries
#
# The outputs of this layer fit naturally with the package's shared data
# contracts such as ``BatchFrameTransform`` and ``BatchJointsState``.


# %%
# Controllers Convert Goals Into Commands
# ---------------------------------------
#
# The ``controllers`` package turns task-space goals into executable command
# outputs. ``IKControllerBase`` defines the common shape: set a goal, then
# calculate the joint-space command needed to move toward it.
#
# ``DifferentialIKController`` builds on this idea for Jacobian-based control.
# It is appropriate when a caller wants to keep issuing end-effector pose or
# delta-pose targets while solving the required joint motion online.
#
# Conceptually:
#
# - kinematics tells you how the robot is structured
# - controller logic tells you how to move from the current state toward a goal
# - the environment or action manager decides how to apply the result


# %%
# The Runtime Flow
# ----------------
#
# A common flow through these modules looks like this:
#
# 1. The environment produces observations and current robot state.
# 2. A policy or planner chooses a target in task space or joint space.
# 3. Kinematics and controllers interpret that target against the current
#    robot structure.
# 4. The resulting joint or end-effector command is forwarded into the action
#    handling path of the environment.
#
# .. code-block:: text
#
#    observation -> policy / planner -> target pose
#                 -> controller / kinematics -> joint command
#                 -> environment action manager -> simulator or robot
#
# This is why control and kinematics belong in ``system_overview``: they sit
# between decision making and execution, and they explain how high-level goals
# become concrete commands.


# %%
# Minimal Differential IK Example
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A low-level solver example is a good executable starting point because it
# does not require a simulator or robot model file. It still shows the real
# controller-side contract: pose error in, joint delta out.

import torch

from robo_orchard_core.controllers.differential_ik import (
    DifferentialIKSolverConfig,
)

solver_cfg = DifferentialIKSolverConfig(
    ik_method="pinv",
    ik_params={"k_val": 1.0},
)
solver = solver_cfg()

delta_pose = torch.tensor([[0.1, -0.05, 0.02]], dtype=torch.float32)
jacobian = torch.eye(3, dtype=torch.float32).unsqueeze(0)

delta_joint = solver.calculate_delta_joint_pos(
    delta_pose=delta_pose,
    jacobian=jacobian,
)

print(delta_joint.shape)
print(delta_joint)


# %%
# Choosing The Right Boundary
# ---------------------------
#
# Use the kinematics layer when you need to:
#
# - inspect or compute transforms from a robot description
# - reason about frame structure independent of a specific environment backend
# - reuse robot-structure queries across simulation and real hardware
#
# Use the controller layer when you need to:
#
# - issue end-effector goals instead of raw joint targets
# - swap IK strategies without rewriting the environment
# - keep motion solving reusable across different runtime backends
#
# Keep action application in environments or managers when you need to:
#
# - validate command shapes against the current task
# - apply commands to a specific simulator or device
# - coordinate command timing with reset/step lifecycle rules


# %%
# Relationship To Other Tutorials
# -------------------------------
#
# This tutorial complements, rather than replaces, the rest of the docs:
#
# - the :doc:`Data Types tutorials </build/gallery/tutorials/datatypes/index>`
#   explain the tensor-backed containers used to move transforms and joint
#   state around the stack
# - :doc:`Environment and Managers </build/gallery/tutorials/system_overview/nonb-02_env_and_managers>`
#   explains where commands are applied in the environment lifecycle
# - the API reference covers the concrete controller and kinematics classes in
#   detail once the runtime role of each layer is clear

# .. note::
#
#    Loading full robot chains from URDF, SDF, or MJCF belongs to the
#    optional ``kinematic`` extra. This tutorial keeps the executable example
#    on the controller side so the docs can demonstrate the motion-command
#    boundary without depending on a robot asset file.


# %%
# Where To Continue
# -----------------
#
# - Return to :doc:`Environment and Managers </build/gallery/tutorials/system_overview/nonb-02_env_and_managers>` for the step/reset lifecycle that eventually applies these commands.
# - Use the :doc:`Data Types tutorials </build/gallery/tutorials/datatypes/index>` if you want the lower-level transform and joint-state containers that move through this stack.
# - Use the :doc:`API reference </autoapi/index>` for concrete controller and kinematics classes once the architectural boundary is clear.
