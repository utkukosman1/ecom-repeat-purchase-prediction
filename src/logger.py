"""Logging setup shared by notebooks and modules.

Convention for this project: use `logger.info` rather than `print`, so that output
carries a timestamp and a source name, and so that the same code logs identically
whether it runs in a notebook, a script or the API.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes timestamped lines to stdout.

    Safe to call repeatedly with the same name. Re-running a notebook cell would
    otherwise attach a second handler and print every line twice, so an existing
    handler is reused rather than replaced.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)

    # Without this, records also reach the root logger and appear twice in Jupyter.
    logger.propagate = False
    return logger
