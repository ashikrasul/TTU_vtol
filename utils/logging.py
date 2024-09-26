import sys

from loguru import logger as log

FORMAT = (
    "<green>{time:MM-DD HH:mm:ss.S}</green> | "
    "<level>{level.icon}</level> | "
    "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

def set_logger(level):
    log.remove()
    log.add(sys.stdout, format=FORMAT, level=level)

    levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    for level_name in levels:
        log.level(level_name, icon=level_name[0])