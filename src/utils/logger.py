import re
import os
import time
import logging
from typing import Literal
from logging.handlers import RotatingFileHandler
from datetime import timedelta

__all__ = ["init_logger"]


class LogFormatter(logging.Formatter):
    color_dict = {
        "DEBUG": "\033[0;37m{}\033[0m",
        "INFO": "\033[0;34m{}\033[0m",
        "NOTE": "\033[1;38;5;46m{}\033[0m",
        "WARNING": "\033[1;48;5;220m{}\033[0m",
        "ERROR": "\033[0;30;41m{}\033[0m",
        "CRITICAL": "\033[0;30;45m{}\033[0m",
    }

    def __init__(
        self, exp_name, colorful=False, start_time=None, time_format="%y-%m-%d %H:%M:%S"
    ):
        super().__init__()
        self.exp_name = exp_name
        self.colorful = colorful
        self.start_time = start_time or time.time()
        self.time_format = time_format

    def format(self, record):
        prefixes = [
            self.exp_name,
            record.name.split(".")[-1],
            record.levelname[0],  # D, I, N, W, E, C
            time.strftime(self.time_format),
            str(timedelta(seconds=record.created - self.start_time)),
        ]
        prefix = f"[{'|'.join([str(p) for p in prefixes if str(p).strip()])}]"
        if record.levelname in ["WARNING", "ERROR", "CRITICAL"]:
            path = os.path.relpath(record.pathname, os.getcwd())
            prefix += f" ({path}:{record.lineno})"
        message = record.getMessage() or ""
        # message = message.replace("\n", "\n" + " " * len(prefix + " "))
        message = message.replace("\n", "\n" + " " * 8)
        if self.colorful:
            return (
                self.color_dict.get(record.levelname, "{}").format(prefix)
                + " "
                + message
            )
        else:
            return prefix + " " + re.sub(r"\033\[[\d;]+m", "", message)


def init_logger(
    package_name: str,
    exp_name: str = None,
    log_file: str = None,
    info_level: Literal[
        "debug", "info", "note", "warning", "error", "critical"
    ] = "info",
    file_max_size_MB: float = 50.0,
    file_backup_count: int = 100,
):
    """
    初始化全局日志配置。
    
    配置控制台输出（带颜色）和文件输出（滚动覆盖）。
    添加了一个自定义的日志级别 'NOTE' (level 25)，介于 INFO 和 WARNING 之间，
    用于强调关键信息。

    Args:
        package_name (str): 记录器的名称，通常是包名。
        exp_name (str, optional): 实验名称，将显示在日志前缀中。
        log_file (str, optional): 日志文件路径。如果为 None，则不写入文件。
        info_level (str, optional): 控制台输出的最低日志级别。默认为 'info'。
        file_max_size_MB (float, optional): 单个日志文件的最大大小 (MB)。
        file_backup_count (int, optional): 保留的历史日志文件数量。
    """
    start_time = time.time()

    # Logging level between INFO and WARNING, used for log something important but not unexpected.
    def note(self, message, *args, **kwargs):
        if self.isEnabledFor(25):
            self._log(25, message, args, **kwargs)

    logging.addLevelName(25, "NOTE")
    logging.Logger.note = note
    logging.NOTE = 25

    logger = logging.getLogger(package_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, info_level.upper()))
    console_handler.setFormatter(
        LogFormatter(
            exp_name, colorful=True, start_time=start_time, time_format="%b%d %H:%M:%S"
        )
    )
    logger.addHandler(console_handler)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            mode="a",
            maxBytes=int(file_max_size_MB * 1024 * 1024),
            backupCount=file_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            LogFormatter(
                exp_name,
                colorful=False,
                start_time=start_time,
                time_format="%b%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

