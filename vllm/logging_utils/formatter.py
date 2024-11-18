import logging
import time
import datetime


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None, style="%"):
        # logging.Formatter.__init__(self, fmt, datefmt, style)
        super().__init__(fmt, datefmt, style)  # 使用 super() 调用父类构造器

    def formatTime(self, record, datefmt=None):
        """Ensure time is formatted with microsecond precision."""
        # 如果 datefmt 被指定，使用它格式化时间
        if datefmt:
            dt = datetime.datetime.fromtimestamp(record.created)
            return dt.strftime(datefmt)
        # 默认返回精确到微秒的时间
        dt = datetime.datetime.fromtimestamp(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f".{int(record.msecs):03d}"

    def format(self, record):
        # msg = logging.Formatter.format(self, record)
        msg = super().format(record)    # 调用父类的格式化方法

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg