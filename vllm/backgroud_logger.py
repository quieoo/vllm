# logger_config.py

import logging
from logging.handlers import BufferingHandler
import threading
import time


outputs=[
# execute
    # "OPT Layer",
    # "OPT Decoder forward",
    # "OPTAttention forward",
    "ModelRunner",
    # "Worker ExecuteModel",
# load
    "LLMENGINE Init",
    "GPUExecuter Init",
    "Loader LoadModel",
    "OPT Initialize",
    "AsyncBlockManager",
# attention
    # "VLLM Attention",
    # "Paged Attention",
    # "ReuseStore",
    # "Scheduler",
    "XFormers",
]

class BackgroundFileHandler(BufferingHandler):
    def __init__(self, filename, maxBytes=1024*1024, flush_interval=0.1):
        super().__init__(maxBytes)
        self.filename = filename
        self.flush_interval = flush_interval
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._flush_periodically)
        self.thread.daemon = True
        self.thread.start()

    def _flush_periodically(self):
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval)
            self.flush()

    def emit(self, record):
        # super().emit(record)
        log_message = self.format(record)
        
        if any(output_str in log_message for output_str in outputs):
            super().emit(record)

    def close(self):
        self._stop_event.set()
        self.thread.join()
        super().close()

    def flush(self):
        if len(self.buffer) > 0:
            with open(self.filename, 'a') as f:
                for record in self.buffer:
                    f.write(self.format(record) + '\n')
            self.buffer = []

class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # 返回包含微秒的时间格式
        ct = self.converter(record.created)
        if datefmt:
            return time.strftime(datefmt, ct)
        else:
            return f"{time.strftime('%m-%d %H:%M:%S', ct)}.{int(record.msecs):03d}{int((record.created * 1000000) % 1000000):06d}"

def setup_logger():
    logger = logging.getLogger("BackgroundLogger")
    logger.setLevel(logging.DEBUG)

    log_file = "/mnt/n0/background_log.txt"
    background_handler = BackgroundFileHandler(log_file)

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = MicrosecondFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    background_handler.setFormatter(formatter)

    logger.addHandler(background_handler)

    return logger

# 设置全局 logger
logger = setup_logger()
