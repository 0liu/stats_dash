# -*- coding: utf-8 -*-

"""
Settings related to China market and other platforms.
"""

import os
from .constants import *


EXCHANGES_DICT = dict(
    CNFutureMD='CNFutureMD',
    CNStockMD='CNStockMD',
    MD='MD',
    aqi='aqi',
    bcReal='5040',
    bcSim='5050',
    citicSim='66666',
    guoDo='9050',
    mk='1025',
    shfe='9999',
    sinoSteel='5300',
    yide='5060',
    zhaoJin='7000',
    ftolSim='4600',
)

AVRO_SCHEMAS_PATH = os.path.dirname(os.path.realpath(__file__)) + "/avro_schemas"


# Alphamongo tick message field name mapping
ALPHAMONGO_TICK_DICT = {
    "createDate": TICK_TIME,
    "symbol": SYMBOL,
    "instrumentName": SYMBOL_NAME,
    "price": LAST_PRICE,
    "openPrice": OPEN_PRICE,
    "highPrice": HIGH_PRICE,
    "lowPrice": LOW_PRICE,
    "closePrice": CLOSE_PRICE,
    "highLimit": HIGH_LIMIT,
    "lowLimit": LOW_LIMIT,
    "settlementPrice": SETTLE,
    "volume": VOLUME,
    "openInterest": OPEN_INTEREST,
    "bid": BID_PRICE,
    "ask": ASK_PRICE,
    "bidVolume": BID_SIZE,
    "askVolume": ASK_SIZE,
    "unitSize": TICKER_DELTA,
    "priceTick": TICK_SIZE,
    "exchangeID": EXCHANGE,
    "longMarginRate": MARGIN_RATIO_LONG,
    "shortMarginRate": MARGIN_RATIO_SHORT,
    "openComissionRate": COMM_OPEN_RATIO,
    "closeComissionRate": COMM_CLOSE_RATIO,
    "closeTodayComissionRate": COMM_CLOSE_TODAY_RATIO,
    "currency": CURRENCY,
}

# Alphamongo database names mapping
ALPHAMONGO_DB_DICT = {
    # Tables
    DB_SPECS: 'instrument',
    DB_MARGIN_COMM: 'instrument_ratio',    
    DB_HIST_1D: 'instrument_history_d',
    DB_HIST_1H: 'instrument_history_h',
    DB_HIST_30M: 'instrument_history_30m',
    DB_HIST_15M: 'instrument_history_15m',
    DB_HIST_5M: 'instrument_history_5m',
    DB_HIST_1M: 'instrument_history_m',
    DB_HIST_1S: None,
    DB_HIST_TICK: 'instrument_history_s',
    # Fields
    'create_date': TICK_TIME,
    'instrument_id': SYMBOL_ID,
    'symbol': SYMBOL,
    'name': SYMBOL_NAME,
    'price': LAST_PRICE,
    'open_price': OPEN_PRICE,
    'high_price': HIGH_PRICE,
    'low_price': LOW_PRICE,
    'close_price': CLOSE_PRICE,
    'high_limit': HIGH_LIMIT,
    'low_limit': LOW_LIMIT,
    'settlement_price': SETTLE,
    'volume': VOLUME,
    'open_interest': OPEN_INTEREST,
    'open_interest_diff': OPEN_INTEREST_DIFF,
    'bid': BID_PRICE,
    'ask': ASK_PRICE,
    'bid_volume': BID_VOLUME,  # not actually used
    'ask_volume': ASK_VOLUME,  # not actually used
    'type': TICKER_TYPE,
    'unit_size': TICKER_DELTA,
    'price_tick': TICK_SIZE,
    'market_open_time': TRADING_HOURS,
    'exchange_id': EXCHANGE,
    'expired': FUT_EXPIRED,
    'open_date': FUT_START_DATE,
    'expire_date': FUT_EXP_DATE,
    'account_id': ACCOUNT_ID,
    'margin_ratio_type': MARGIN_RATIO_TYPE,
    'long_margin_ratio': MARGIN_RATIO_LONG,
    'short_margin_ratio': MARGIN_RATIO_SHORT,
    'commission_ratio_type': COMM_RATIO_TYPE,
    'open_ratio': COMM_OPEN_RATIO,
    'close_ratio': COMM_CLOSE_RATIO,
    'close_today_ratio': COMM_CLOSE_TODAY_RATIO,
    'max_limit_order_volume': MAX_ORDER_SIZE,
    'min_limit_order_volume': MIN_ORDER_SIZE,
}
ALPHAMONGO_DB_DICT.update(dict([reversed(i) for i in ALPHAMONGO_DB_DICT.items()]))  # Add reverse lookup

ALPHAMONGO_DB_TICKTIME_FORMAT = {
    K_DAY_FMT: "%Y-%m-%d 00:00:00",
    K_HOUR_FMT: "%Y-%m-%d %H:%M:%S",
    K_MIN_FMT: "%Y-%m-%d %H:%M:%S",
    K_SEC_FMT: "%Y-%m-%d %H:%M:%S.%f",
    TICK_FMT: "%Y-%m-%d %H:%M:%S.%f",
}

INVALID_DATES = [
    '2019-05-05',
]