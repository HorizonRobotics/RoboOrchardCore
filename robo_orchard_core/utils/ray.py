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
from typing import Any

import ray
import torch
from ray.actor import ActorProxy
from ray.exceptions import GetTimeoutError, RayActorError
from typing_extensions import Generic, TypeVar

from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType,
    Config,
    ConfigInstanceOf,
)
from robo_orchard_core.utils.logging import LoggerManager

logger = LoggerManager().get_child(__name__)


__all__ = [
    "DEFAULT_RAY_INIT_CONFIG",
    "is_ray_actor_alive",
    "RayActorNotAliveError",
    "RayActorDiedError",
    "RayRemoteClassConfig",
    "RayRemoteInstanceConfig",
    "RayRemoteInstance",
    "ray_init",
]

DEFAULT_RAY_INIT_CONFIG = {
    "address": None,
    "num_cpus": None,
    "num_gpus": None,
    "resources": None,
    "object_store_memory": None,
    "local_mode": False,
    "ignore_reinit_error": False,
    "include_dashboard": None,
    "dashboard_port": None,
    "job_config": None,
    "configure_logging": True,
    "logging_level": "info",
    "logging_format": None,
    "log_to_driver": True,
    "namespace": None,
    "runtime_env": None,
}


def ray_init(ray_init_config: dict[str, Any] | None = None):
    """Initialize Ray with the given configuration.

    In multi-process scenarios, this function still may create multiple
    ray instances because ray.is_initialized() seems to not be process-safe.
    If in such scenarios, please start ray from command line!

    """
    if not ray.is_initialized():
        if ray_init_config is not None:
            ray.init(**ray_init_config)
        else:
            ray.init(**DEFAULT_RAY_INIT_CONFIG)


class RayActorDiedError(Exception):
    """Exception raised when a Ray actor has died unexpectedly."""

    pass


class RayActorNotAliveError(Exception):
    """Exception raised when a Ray actor is not alive."""

    pass


def is_ray_actor_alive(
    remote, timeout: int = 10, error_info: dict | None = None
) -> bool:
    """Check if a Ray actor is alive within a timeout period.

    If the actor is died, it raises a RayActorDiedError.
    """

    try:
        ray.get(remote.__ray_ready__.remote(), timeout=timeout)
        if error_info is not None:
            error_info["error"] = "State: ALIVE"
        return True
    except GetTimeoutError:
        if error_info is not None:
            error_info["error"] = (
                f"Timed out after {timeout} seconds waiting for actor ready."
            )
        return False
    except RayActorError as e:
        err_msg = f"Ray actor is dead. cause: {e}"
        if error_info is not None:
            error_info["error"] = err_msg
        raise RayActorDiedError(err_msg) from e
    except Exception as e:
        if error_info is not None:
            error_info["error"] = str(e)
        return False


class RayRemoteClassConfig(Config):
    __exclude_config_type__: bool = True

    num_cpus: float | int | None = 0.5
    """Number of CPU cores to allocate to the remote actor."""
    num_gpus: float | int | None = 0.1 if torch.cuda.is_available() else None
    """Number of GPUs to allocate to the remote actor."""
    memory: int = 1 * 1024**3
    """heap memory request in bytes. Default to 1GB."""
    runtime_env: dict[str, Any] | None = None
    """The runtime environment for the remote actor.

    See https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#concepts
    """


T = TypeVar("T")

ClassConfigType_co = TypeVar(
    "ClassConfigType_co", bound=ClassConfig, covariant=True
)


class RayRemoteInstanceConfig(ClassConfig[T], Generic[T, ClassConfigType_co]):
    """Configuration for creating a Ray remote instance.

    Template parameters:
        T: The type of the remote instance.
        ClassConfigType_co: The type of the configuration for the class to be instantiated remotely.
    """  # noqa: E501

    class_type: ClassType[T]

    remote_class_config: RayRemoteClassConfig = RayRemoteClassConfig()
    """The configuration for the remote class."""

    ray_init_config: dict[str, Any] | None = None
    """The configuration for initializing Ray. If None, use default."""

    check_init_timeout: int = 60
    """Timeout in seconds for checking if the remote actor is initialized."""

    instance_config: ConfigInstanceOf[ClassConfigType_co]


class RayRemoteInstance(Generic[T]):
    """A class that manages a Ray remote actor instance."""

    cfg: RayRemoteInstanceConfig["RayRemoteInstance", ClassConfig[T]]

    remote_cls: Any
    """The Ray remote class."""

    _remote: ActorProxy[T]
    """The Ray remote actor instance."""

    def __init__(self, cfg: RayRemoteInstanceConfig, **kwargs):
        self.cfg = cfg

        ray_init(self.cfg.ray_init_config)

        remote_cls = ray.remote(**self.cfg.remote_class_config.model_dump())(
            self.cfg.instance_config.class_type
        )
        self.remote_cls = remote_cls

        remote = remote_cls.remote(self.cfg.instance_config, **kwargs)  # type: ignore
        self._remote: ActorProxy[T] = remote
        self._remote_checked = False

    @property
    def remote(self) -> ActorProxy[T]:
        """Get the Ray remote actor instance.

        Raises:
            RayActorNotAliveError: If the Ray actor is not alive.
        """

        if self._remote_checked:
            return self._remote

        ray_error_info = {}
        if not is_ray_actor_alive(
            self._remote,
            timeout=self.cfg.check_init_timeout,
            error_info=ray_error_info,
        ):
            try:
                ray.kill(self._remote)
                self._remote = None  # type: ignore
            except Exception:
                pass
            raise RayActorNotAliveError(
                f"Ray actor failed to be alive within "
                f"{self.cfg.check_init_timeout} seconds. "
                f"Reason: {ray_error_info}"
                "Please check the ray remote class config and cluster that "
                "enough resources "
                f"are available: {ray.available_resources()}"
            )
        self._remote_checked = True
        return self._remote

    def close(self, remote_close_method: str | None = None):
        """Release the managed Ray actor instance."""
        if not hasattr(self, "_remote") or self._remote is None:
            return

        remote = self._remote
        if remote_close_method is not None:
            try:
                ray.get(getattr(remote, remote_close_method).remote())
            except Exception:
                pass

        try:
            ray.kill(remote)
        except Exception:
            pass

        self._remote = None  # type: ignore
        self._remote_checked = False
