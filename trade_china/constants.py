# -*- coding: utf-8 -*-

"""
Global constants.
"""


# Historical database table names
DB_SPECS = 'DB_SPECS'
DB_MARGIN_COMM = 'DB_MARGIN_COMM'
DB_HIST_1D = 'DB_HIST_1D'
DB_HIST_1H = 'DB_HIST_1H'
DB_HIST_30M = 'DB_HIST_30M'
DB_HIST_15M = 'DB_HIST_15M'
DB_HIST_5M = 'DB_HIST_5M'
DB_HIST_1M = 'DB_HIST_1M'
DB_HIST_1S = 'DB_HIST_1S'
DB_HIST_TICK = 'DB_HIST_TICK'

# Historical database and tick data field names
TICK_TIME = 'TICK_TIME'
SYMBOL_ID = 'SYMBOL_ID'  # System internal ID for a symbol/contract
SYMBOL = 'SYMBOL'
SYMBOL_NAME = 'SYMBOL_NAME'
LAST_PRICE = 'LAST_PRICE'
OPEN_PRICE = 'OPEN_PRICE'
HIGH_PRICE = 'HIGH_PRICE'
LOW_PRICE = 'LOW_PRICE'
CLOSE_PRICE = 'CLOSE_PRICE'
HIGH_LIMIT = 'HIGH_LIMIT'  # high limit price
LOW_LIMIT = 'LOW_LIMIT'  # low limit price
SETTLE = 'SETTLE'  # settlement price
VOLUME = 'VOLUME'
OPEN_INTEREST = 'OPEN_INTEREST'
OPEN_INTEREST_DIFF = 'OPEN_INTEREST_DIFF'  # difference between two consecutive time stamps
BID_PRICE = 'BID_PRICE'
ASK_PRICE = 'ASK_PRICE'
BID_SIZE = 'BID_SIZE'  # Quote size on bid price
ASK_SIZE = 'ASK_SIZE'  # Quote size on ask price
BID_VOLUME = 'BID_VOLUME'  # Traded volume on bid price
ASK_VOLUME = 'ASK_VOLUME'  # Trade volume on ask price
TICKER_TYPE = 'TICKER_TYPE'  # security type: futures, stocks, etc.
TICKER_DELTA = 'TICKER_DELTA'  # Contract unit size
TICK_SIZE = 'TICK_SIZE'  # Minimum price fluctuation
TRADING_HOURS = 'TRADING_HOURS'
EXCHANGE = 'EXCHANGE'
FUT_EXPIRED = 'FUT_EXPIRED'  # If a futures contract expired
FUT_START_DATE = 'FUT_START_DATE'
FUT_EXP_DATE = 'FUT_EXP_DATE'
ACCOUNT_ID = 'ACCOUNT_ID'  # System account ID, not broker ID
MARGIN_RATIO_TYPE = 'MARGIN_RATIO_TYPE'  # By money or By volume
MARGIN_RATIO_LONG = 'MARGIN_RATIO_LONG'
MARGIN_RATIO_SHORT = 'MARGIN_RATIO_SHORT'
COMM_RATIO_TYPE = 'COMM_RATIO_TYPE'  # Commission ratio type. By money or By volume
COMM_OPEN_RATIO = 'COMM_OPEN_RATIO'
COMM_CLOSE_RATIO = 'COMM_CLOSE_RATIO'
COMM_CLOSE_TODAY_RATIO = 'COMM_CLOSE_TODAY_RATIO'
MAX_ORDER_SIZE = 'MAX_ORDER_SIZE'
MIN_ORDER_SIZE = 'MIN_ORDER_SIZE'
CURRENCY = 'CURRENCY'

# Historical database time stamp format names
K_DAY_FMT = 'K_DAY_FMT'
K_HOUR_FMT = 'K_HOUR_FMT'
K_MIN_FMT = 'K_MIN_FMT'
K_SEC_FMT = 'K_SEC_FMT'
TICK_FMT = 'TICK_FMT'


# DataFrame column names for pair defs, specs and studies
PAIR_NAME = 'pair_name'
PAIR_CAPITAL = 'pair_capital'
PAIR_VAR = 'pair_var'
LEG_i_SYMBOL = 'leg{i}_symbol'  # Leg counting i starts from 1
LEG_i_N = 'leg{i}_N'
LEG_i_TICKER_DELTA = 'leg{i}_ticker_delta'
LEG_i_HIGH_LIMIT = 'leg{i}_high_limit'
LEG_i_LOW_LIMIT = 'leg{i}_low_limit'
LEG_i_SETTLE = 'leg{i}_settle'
LEG_i_MARGIN_RATIO_LONG = 'leg{i}_margin_ratio_long'
LEG_i_MARGIN_RATIO_SHORT = 'leg{i}_margin_ratio_short'
LEG_i_COMM_RATIO_TYPE = 'leg{i}_comm_ratio_type'
LEG_i_COMM_RATIO_OPEN = 'leg{i}_comm_ratio_open'
LEG_i_COMM_RATIO_CLOSE = 'leg{i}_comm_ratio_close'
LEG_i_COMM_RATIO_CLOSE_TODAY = 'leg{i}_comm_ratio_close_today'
LEG_i_PRICE = 'leg{i}_price'
LEG_i_DAY_CHANGE = 'leg{i}_day_change'
LEG_i_DAY_CHANGE_N = 'leg{i}_day_change_N'  # N is number of contracts
LEG_i_DAY_CHANGE_m_DAY = 'leg{i}_day_change_{m}d'  # Running period counting m starts from 1
LEG_i_DAY_CHANGE_N_m_DAY = 'leg{i}_day_change_N_{m}d'
PAIR_VAL = 'pair_val'
PAIR_VAL_DAY_HIGH = 'pair_val_day_high'
PAIR_VAL_DAY_LOW = 'pair_val_day_low'
PAIR_DAY_CHANGE_RATIO_m_DAY = 'pair_day_change_ratio_{m}d'
PAIR_VAL_LEVEL_MINUS_x = 'pair_val_level_minus_{x}'  # Level counting x starts from 1
PAIR_VAL_LEVEL_PLUS_x = 'pair_val_level_plus_{x}'
PAIR_VAL_LEVEL_MIN_DIST_m_DAY = 'pair_val_level_min_dist_{m}d'
PAIR_VAL_LEVEL_MAX_DIST_m_DAY = 'pair_val_level_max_dist_{m}d'
PAIR_VAL_LEVEL_MIN_DIST = 'pair_val_level_min_dist'
PAIR_VAL_LEVEL_MIN_DIST_SCALE = 'pair_val_level_min_distance_scale'
PAIR_VAL_LEVEL_MAX_DIST = 'pair_val_level_max_distance'
LEG_i_PREV_CLOSE = 'leg{i}_prev_close'
PAIR_VAL_PREV_CLOSE = 'pair_val_prev_close'
PAIR_VAL_PREV_CLOSE_OFFSET = 'pair_val_prev_close_offset'
PAIR_TOTAL_MARGIN = 'pair_total_margin'
PAIR_TOTAL_NUMBER = 'pair_total_number'
PAIR_NUMBER_PER_LEVEL = 'pair_number_per_level'
PAIR_VAL_HIGH_m_DAY = 'pair_val_high_{m}d'
PAIR_VAL_LOW_m_DAY = 'pair_val_low_{m}d'
PAIR_VAL_MA_m_DAY = 'pair_val_ma_{m}d'
PAIR_CORR_m_DAY = 'pair_corr_{m}d'
LEG_i_ATR_m_DAY = 'leg{i}_atr_{m}d'
PAIR_ATR_m_DAY = 'pair_atr_{m}d'
PAIR_HL_ATR_RATIO_m_DAY = 'pair_hl_atr_ratio_{m}d'
PAIR_HL_MIN_RISK_RATIO_m_DAY = 'pair_hl_min_risk_ratio_{m}d'

# DataFrame column names for single contract defs, specs and studies. May overlap with database fields.
TICKER = 'TICKER'
SYMBOL_VAR = 'symbol_val_at_risk'
MID_PRICE = 'MID_PRICE'
MID_PRICE_DAY_HIGH = 'mid_price_day_high'
MID_PRICE_DAY_LOW = 'mid_price_day_low'
MARGIN_VAL = 'margin_value'
PREV_CLOSE = 'prev_close'
PREV_VOLUME = 'prev_volume'
SYMBOL_VAL = 'symbol_val'
SYMBOL_PREV_VAL = 'symbol_prev_val'
SYMBOL_VAL_DAY_CHANGE = 'symbol_val_day_change'
SYMBOL_VAL_DAY_CHANGE_m_DAY = 'symbol_val_day_change_{m}d'
PREV_OPEN_INTEREST = 'prev_open_interest'
OPEN_INTEREST_TICK_CHANGE = 'open_interest_tick_change'
PREV_OPEN_INTEREST_CAPITAL = 'prev_open_interest_capital'
SYMBOL_VAL_LEVEL_MINUS_x = 'symbol_val_level_minus_{x}'
SYMBOL_VAL_LEVEL_PLUS_x = 'symbol_val_level_PLUS_{x}'
SYMBOL_TOTAL_NUMBER = 'symbol_total_number'
SYMBOL_NUMBER_PER_LEVEL = 'symbol_number_per_level'
SYMBOL_PRICE_HIGH_m_DAY = 'symbol_price_high_{m}d'
SYMBOL_PRICE_LOW_m_DAY = 'symbol_price_low_{m}d'
SYMBOL_PRICE_MA_m_DAY = 'symbol_price_ma_{m}d'
SYMBOL_ATR_m_DAY = 'symbol_atr_{m}d'
SYMBOL_HL_ATR_RATIO_m_DAY = 'symbol_hl_atr_ratio_{m}d'
ROLL_RTN = 'ROLL_RTN'