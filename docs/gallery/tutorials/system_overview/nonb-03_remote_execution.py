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

"""Remote Execution with Ray
=========================================
"""

# %%
# Why Remote Wrappers Exist
# -------------------------
#
# The local interfaces in ``robo_orchard_core`` are intentionally simple:
# policies expose ``act`` and ``reset``; environments expose ``reset``,
# ``step``, ``rollout``, and ``close``. When those same abstractions need to
# run in another process or on another machine, the package keeps the local
# interface shape and moves the transport concern into dedicated wrappers.
#
# This is the role of the Ray integration layer:
#
# - ``PolicyConfig.as_remote()`` converts a local policy config into a
#   ``RemotePolicyConfig``.
# - ``EnvBaseCfg.as_remote()`` converts a local environment config into a
#   ``RemoteEnvCfg``.
# - ``RayRemoteClassConfig`` describes actor resources such as CPU, GPU, and
#   Ray runtime environment requirements.
#
# The architectural point is that *domain logic stays local* while *execution
# placement becomes configurable*.


# %%
# When To Read This Tutorial
# --------------------------
#
# Read this tutorial when you already understand the local environment or
# policy interfaces and need to answer questions such as:
#
# - how to move the same config-driven runtime object behind a Ray actor
# - where synchronous calls stop and future-based coordination begins
# - which responsibilities belong in transport wrappers instead of domain code


# %%
# The Config Flow
# ---------------
#
# A common remote execution flow looks like this:
#
# 1. Define the same local config you would use for an in-process policy or
#    environment.
# 2. Wrap it with ``as_remote(...)`` when you need the runtime object to live
#    behind a Ray actor.
# 3. Instantiate the returned remote config and keep interacting through the
#    normal policy or environment methods.
#
# The same pattern applies to environments.
#
# This keeps the construction pattern consistent with the rest of the package:
# config first, runtime object second.


# %%
# Minimal Config-To-Remote Example
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The snippet below uses the real config APIs but stops before starting any
# Ray actor. That keeps the example lightweight while still showing the
# concrete wrapping flow users call in practice.

import gymnasium as gym
import torch

from robo_orchard_core.envs.env_base import EnvBase, EnvBaseCfg, EnvStepReturn
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import ClassType
from robo_orchard_core.utils.ray import RayRemoteClassConfig


class DemoPolicy(PolicyMixin[torch.Tensor, torch.Tensor]):
    def reset(self, *args, **kwargs):
        return None

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + 1


class DemoPolicyConfig(PolicyConfig[DemoPolicy]):
    class_type: ClassType[DemoPolicy] = DemoPolicy


class DemoEnv(EnvBase[torch.Tensor, float]):
    @property
    def num_envs(self) -> int:
        return 1

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def reset(self, seed=None, env_ids=None, **kwargs):
        return torch.zeros(1), {}

    def step(self, action):
        return EnvStepReturn(
            observations=action,
            rewards=0.0,
            terminated=False,
            truncated=False,
            info={},
        )

    def close(self):
        return None


class DemoEnvCfg(EnvBaseCfg[DemoEnv]):
    class_type: ClassType[DemoEnv] = DemoEnv


remote_policy_cfg = DemoPolicyConfig().as_remote(
    remote_class_config=RayRemoteClassConfig(num_cpus=1, num_gpus=None)
)
remote_env_cfg = DemoEnvCfg().as_remote(check_init_timeout=5)

print(type(remote_policy_cfg).__name__)
print(type(remote_env_cfg).__name__)
print(remote_policy_cfg.remote_class_config.num_cpus)
print(remote_env_cfg.check_init_timeout)


# %%
# Synchronous and Asynchronous Boundaries
# ---------------------------------------
#
# The remote wrappers expose both synchronous and asynchronous interfaces:
#
# - ``RemotePolicy.act`` blocks until the remote actor returns an action.
# - ``RemotePolicy.async_act`` returns a future when the caller wants to
#   overlap work.
# - ``RemoteEnv.step`` and ``RemoteEnv.reset`` provide blocking environment
#   calls.
# - ``RemoteEnv.async_step``, ``async_reset``, and ``async_rollout`` provide
#   future-based control when the caller coordinates multiple workers.
#
# The important design choice is that the API shape stays close to the local
# version. A caller can often upgrade from local to remote execution by
# changing the config path instead of rewriting the control flow from scratch.


# %%
# What Belongs In Remote Wrappers
# -------------------------------
#
# Remote wrappers are responsible for:
#
# - actor lifecycle and readiness checks
# - Ray resource requests and initialization options
# - translating method calls to actor invocations
# - preserving familiar policy and environment interfaces for callers
#
# Remote wrappers should *not* become the place where domain behavior is
# reimplemented. The wrapped policy still owns action generation, and the
# wrapped environment still owns reset, step, and rollout semantics.


# %%
# When To Use This Layer
# ----------------------
#
# Use the remote execution layer when you need one or more of the following:
#
# - running multiple environments or policies in parallel across processes
# - moving GPU-heavy inference off the caller thread or process
# - scaling the same policy or environment abstraction across Ray workers
# - keeping the local ``EnvBase`` or ``PolicyMixin`` contract while changing
#   execution placement
#
# Stay with local objects when a single-process runtime is easier to debug,
# faster to iterate on, or all you need for the current experiment.


# %%
# How It Fits With The Rest Of The Package
# ----------------------------------------
#
# The manager-based environment loop described in the system overview remains
# the conceptual center of the package. Remote execution sits outside that
# loop as an integration concern:
#
# - configuration decides whether objects are local or remote
# - policy and environment contracts stay the same
# - actor boundaries are isolated in ``policy.remote``, ``envs.remote``, and
#   ``utils.ray``
#
# This is why remote execution belongs in ``system_overview``: it changes how
# a system is deployed and coordinated, not how the data model itself is
# represented.


# %%
# Where To Continue
# -----------------
#
# - Return to :doc:`Environment and Managers </build/gallery/tutorials/system_overview/nonb-02_env_and_managers>` if you want to review the local runtime loop that remote wrappers preserve.
# - Move to :doc:`CLI Tools and Extensions </build/gallery/tutorials/system_overview/nonb-05_tools_and_extensions>` when the next question is how operators or services trigger those runtime objects.
# - Use the :doc:`API reference </autoapi/index>` for the concrete ``RemoteEnv``, ``RemotePolicy``, and ``RayRemoteClassConfig`` APIs.
