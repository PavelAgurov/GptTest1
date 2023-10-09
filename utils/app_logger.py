"""
    Custom logger for application
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121

import os
import logging
from datetime import datetime

from utils.colored_console_formatter import ColoredConsoleFormatter

LOG_FOLDER = '.logs'

def init_root_logger():
    """Init root logger"""

    logger = logging.getLogger()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColoredConsoleFormatter())
    logger.addHandler(stream_handler)
    os.makedirs(LOG_FOLDER, exist_ok=True)
    file_handler = logging.FileHandler(rf'{LOG_FOLDER}\log{datetime.now().strftime("%Y-%m-%d")}.log')
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    return logger
