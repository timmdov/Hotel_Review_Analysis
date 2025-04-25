import logging

from colorlog import ColoredFormatter


def get_logger(name: str = None) -> logging.Logger:
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s : %(levelname)-8s : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "light_red",
            "CRITICAL": "bold_red"
        }
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
