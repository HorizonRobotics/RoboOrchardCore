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

"""The logging module."""

import logging

from typing_extensions import Dict, Optional, Self

DEFAULT_LOG_FORMAT = (
    "%rank %(asctime)-15s %(levelname)s "
    "| %(process)d | %(threadName)s | "
    "%(name)s:L%(lineno)d %(message)s"
)


def wrap_log_fmt_with_rank(format: str) -> str:
    """Wrap the log format with the rank of the process."""
    from robo_orchard_core.utils.distributed import get_dist_info

    if "%rank" in format:
        dist_info = get_dist_info()
        if dist_info.world_size > 1:
            format = format.replace(
                "%rank",
                "Rank[{}/{}]".format(dist_info.rank, dist_info.world_size),
            )
        else:
            # remove %rank if not in distributed mode
            format = format.replace("%rank ", "")
    return format


def singleton(cls):
    """A singleton decorator for a class."""
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class LoggerManager:
    """A logger manager that manages the logger.

    This class is a singleton class that manages the logger. It provides
    methods to get the logger and its child logger. You can use the logger
    manager to handle all loggers in the project.

    Note that this class is a singleton class, so if you create multiple
    instances of this class, only the first instance will be used! Passing
    different arguments to the constructor will have no effect after the
    first instantiation.

    Example:

    .. code-block:: python

        # To get a child logger
        from robo_orchard_core.utils.logging import LoggerManager

        logger = LoggerManager().get_child(__name__)


    Args:
        format (str, optional): The format of the log. Defaults to
            DEFAULT_LOG_FORMAT.
        level (int, optional): The level of the log. Defaults to logging.INFO.
        handlers (Optional[list[logging.Handler]], optional): The handlers of
            the log. Defaults to None.

    """

    def __init__(
        self,
        format: str = DEFAULT_LOG_FORMAT,
        level: int = logging.INFO,
        handlers: Optional[list[logging.Handler]] = None,
        **kwargs,
    ):
        format = wrap_log_fmt_with_rank(format)
        self._logger = logging.getLogger("LoggerManager")
        self._logger.setLevel(level)
        self._format = format
        self._level = level

        if handlers is None:
            handlers = [
                logging.StreamHandler(),
            ]
        self.set_handlers(handlers)
        self.set_format(format)
        self._child_loggers: Dict[str, logging.Logger] = {}

    def get_logger(self) -> logging.Logger:
        """Get the global logger."""
        return self._logger

    def get_child(
        self,
        name: str,
    ) -> logging.Logger:
        """Get the child logger.

        Args:
            name (str): The name of the child logger.

        Returns:
            logging.Logger: The child logger.

        """
        if name in self._child_loggers:
            return self._child_loggers[name]
        ret = self._logger.getChild(name)
        self._child_loggers[name] = ret
        return ret

    def set_level(self, level: int) -> Self:
        """Set the level of the logger.

        Args:
            level (int): The level of the logger.

        Returns:
            Self: The logger manager.

        """

        self._level = level

        loggers_to_set = [self._logger]

        for logger in loggers_to_set:
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)

        return self

    def set_format(
        self,
        format: str,
    ) -> Self:
        """Set the format of the logger.

        This method set the format of all handlers of the logger.

        Args:
            format (str): The format of the logger.

        Returns:
            Self: The logger manager.

        """
        format = wrap_log_fmt_with_rank(format)
        self._format = format

        loggers_to_set = [self._logger]

        for logger in loggers_to_set:
            for handler in logger.handlers:
                handler.setFormatter(logging.Formatter(format))

        return self

    def set_handlers(self, handlers: list[logging.Handler]) -> Self:
        """Set the handlers of the logger.

        Args:
            handlers (list[logging.Handler]): The handlers to set.

        Returns:
            Self: The logger manager.
        """

        loggers_to_set = [self._logger]

        for logger in loggers_to_set:
            logger.handlers.clear()
            for handler in handlers:
                handler.setFormatter(logging.Formatter(self._format))
                handler.setLevel(self._level)
                logger.addHandler(handler)
        return self
