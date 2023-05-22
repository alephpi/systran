import logging
import os


def init_logger(level=None):
    if level is None:
        level = "INFO"

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d000 UTC [%(module)s@%(processName)s] %(levelname)s %(message)s",
        datefmt="%Y-%b-%d %H:%M:%S",
        level=os.getenv("LOG_LEVEL", level),
    )


def get_logger():
    return logging.getLogger("pytorch_transformer")
