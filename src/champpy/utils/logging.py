import logging
import sys

LOG_FORMAT = "[%(asctime)s - %(levelname)s - %(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """
    Set up a Basic logger that will be configured when importing the library.

    Returns:
        None
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=True,
    )
