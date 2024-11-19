import logging


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} - [{levelname}] - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    return logging.getLogger(__name__)
