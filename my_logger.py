# Logging the way we like it.

import logging
from logging import StreamHandler,FileHandler
from logging.handlers import RotatingFileHandler
import sys

"""
This module generates log files that have a max length of 50K bytes.
It uses RotatingFileHandler and creates no more than 3 backup files;
that means that when the file reaches 50K bytes, it is renamed and
a new logfile is opened.
Exampe: you name the file "myapp.log". Backup files will be named:
myapp.log.1, myapp.log.2, myapp.log.3. You can change the number of backups
by editing this code.

Explanation of logger levels.
Specify level=logging.CRITICAL, etc.
Each level includes messages from the levels before it. So specifying WARNING will also
output CRITICAL and ERROR messages to the logfile.
DEBUG is most wordy and should be turned off in production runs.
Print to log file using logger.info('message') or logger.debug or logger.warning, etc.
When level=logging.INFO, any time the code runs logger.debug, you will not get that message.
"""

#rlogger = logging.getLogger().setLevel(logging.NOTSET)      # root logger
defFormat = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
defFormatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
# must sett root logger defaults before overriding with specific handlers
# This logger is used for running analysis programs, so precise datetime is not needed
logging.basicConfig(level=logging.DEBUG, format=defFormat, datefmt='%m-%d %H:%M')

def setup_logger(name, log_file, formatter=defFormatter, level=logging.INFO):
    """
    Setup as many loggers as you want
    :param name: typically the module name. You should use __name__ unless you have a better idea.
    :param log_file: name of the logging file.
    :param formatter: defines the format for each log line.
    :param level: standard logger levels are CRITICAL, ERROR, WARNING, INFO, DEBUG.
    :return: the logger. Use it like this: logger.info('message')
    """
    #handler = RotatingFileHandler(log_file, maxBytes=100000, backupCount=3)
    fileHandler = FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(level)

    # console will show INFO messages, not DEBUG
    consoleHandler = StreamHandler(sys.stdout)
    consoleFormatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
    consoleHandler.setFormatter(consoleFormatter)
    consoleHandler.setLevel(logging.INFO)

    logging.getLogger().addHandler(fileHandler)
    logging.getLogger().addHandler(consoleHandler)
    logger = logging.getLogger(name)
    #logger.setLevel(level)
    return logger
