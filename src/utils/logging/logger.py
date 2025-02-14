import logging
import os
import sys


class Logger:
    """
    A singleton Logger class to standardize logging across the project.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, name: str = "DeepSampler", level: int = logging.INFO, log_file: str = None
    ):
        # Only initialize once
        if hasattr(self, "initialized") and self.initialized:
            return

        # Create a logger with the specified name and level.
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Define a common formatter.
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Create and add a console handler.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Optionally add a file handler if log_file is provided.
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.initialized = True

    def get_logger(self) -> logging.Logger:
        """
        Return the configured logger instance.
        """
        return self.logger
