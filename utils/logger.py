"""
logger setup.
Create On 3rd Dec, 2019
Author: Bohang Li
"""
import os
import logging
import logging.handlers


def init_logging(log_dir=None, log_file='anti-deepfake.log', log_level='info', logSizeMB=10, backupNum=6):
    if log_dir is None:
        log_dir = os.path.abspath("./logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig()
    log = logging.getLogger()
    if log_level == 'debug':
        log.setLevel(logging.DEBUG)
    elif log_level == 'info':
        log.setLevel(logging.INFO)
    elif log_level == 'warning':
        log.setLevel(logging.WARNING)
    elif log_level == 'error':
        log.setLevel(logging.ERROR)
    elif log_level == 'critical':
        log.setLevel(logging.CRITICAL)
    else:
        log.setLevel(logging.FATAL)
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=int(logSizeMB)*1024*1024,
        backupCount=int(backupNum)
    )
    formatter = logging.Formatter('%(levelname)s:%(filename)s:%(lineno)d:%(asctime)s %(message)s', "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    return log
