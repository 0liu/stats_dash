# -*- coding: utf-8 -*-

import os
import logging, coloredlogs
import configparser
import pytz

coloredlogs.DEFAULT_FIELD_STYLES['levelname']['color'] = 'blue'

SHANGHAI = pytz.timezone('Asia/Shanghai')
UTC = pytz.UTC
CONFIG_PATH = os.path.realpath(os.path.expanduser('~/.trade_china'))
CONFIG_FILE = 'trade_china.ini'
if not os.path.exists(CONFIG_PATH):
    os.makedirs(CONFIG_PATH)
CONFIG_FIELDS = {
    'TickData': ('host', 'port', 'user', 'password', 'exchange'),
    'HistData': ('host', 'port', 'user', 'password', 'database'),
    'Trading': ('account_id',),
}


def get_logger(logger_name, file_level=logging.DEBUG, stream_level='INFO'):
    logger = logging.getLogger('logger_name')
    logger.setLevel(file_level)
    # create file handler which logs even debug messages
    logger_file = os.path.join(CONFIG_PATH, logger_name+'.log')
    if os.path.exists(logger_file):
        os.remove(logger_file)
    fh = logging.FileHandler(logger_file)
    fh.setLevel(file_level)
    # create formatter and add it to the handlers
    logger_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(logger_format)
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    # Add colored stream logger    
    coloredlogs.install(level=stream_level, logger=logger, fmt=logger_format)
    return logger

def get_config(logger=None, reset=False):
    if logger is None:
        logger = get_logger('utils')
    config = configparser.ConfigParser()
    config_file = os.path.join(CONFIG_PATH, CONFIG_FILE)
    if not reset and os.path.exists(config_file):
       config.read(config_file)
    config_changed = False
    for section in CONFIG_FIELDS.keys():
        if section not in config:
            config[section] = {}
            config_changed = True
        for field in CONFIG_FIELDS[section]:
            if field not in config[section]:
                print("\nSetting {sec}:{f} not found. Please input:\n".format(sec=section, f=field))
                config[section][field] = input().strip()
                config_changed = True
    if config_changed:
        with open(config_file, 'w') as f:
            config.write(f)
    return config

def get_number_of_decimal(price_tick):
    str_price_tick_ = str(price_tick)
    decimal_pos_ = str_price_tick_.find('.')
    if decimal_pos_ == -1 or str_price_tick_[decimal_pos_ + 1:] == '0':
        return 0
    else:
        return len(str_price_tick_) - (decimal_pos_ + 1)
