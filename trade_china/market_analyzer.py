# -*- coding: utf-8 -*-

"""
Compute statistics and indicators from real-time market data.
"""


from collections import defaultdict
from queue import Empty
from threading import Thread, Lock
from datetime import timedelta
from time import sleep
import numpy as np
import pandas as pd
from .dataport import DataPort
from .utils import get_logger, get_number_of_decimal
from .constants import *


class MarketAnalyzer:
    """
    Stream market data and compute indicators in real-time.
    """
    PAIR_STUDIES = [
        # stats
        PAIR_VAL_PREV_CLOSE,
        PAIR_VAL_HIGH_m_DAY,
        PAIR_VAL_LOW_m_DAY,
        PAIR_VAL_MA_m_DAY,
        PAIR_CORR_m_DAY,
        PAIR_ATR_m_DAY,
        LEG_i_ATR_m_DAY,
        LEG_i_PREV_CLOSE,
        LEG_i_DAY_CHANGE_m_DAY,
        LEG_i_DAY_CHANGE_N_m_DAY,
        PAIR_DAY_CHANGE_RATIO_m_DAY,
        PAIR_VAL_LEVEL_MIN_DIST_m_DAY,
        PAIR_VAL_LEVEL_MAX_DIST_m_DAY,
        PAIR_HL_ATR_RATIO_m_DAY,
        PAIR_HL_MIN_RISK_RATIO_m_DAY,
        # indicators
        LEG_i_DAY_CHANGE,
        LEG_i_DAY_CHANGE_N,
        PAIR_VAL_LEVEL_MAX_DIST,
        PAIR_VAL_LEVEL_MIN_DIST,
        PAIR_VAL_LEVEL_MINUS_x,
        PAIR_VAL_LEVEL_PLUS_x,
        PAIR_TOTAL_NUMBER,
        PAIR_NUMBER_PER_LEVEL,
    ]

    SINGLE_STUDIES = [
        # stats
        MARGIN_VAL,
        PREV_CLOSE,
        SYMBOL_PREV_VAL,
        PREV_VOLUME,
        PREV_OPEN_INTEREST,
        PREV_OPEN_INTEREST_CAPITAL,
        SYMBOL_VAL_DAY_CHANGE_m_DAY,
        SYMBOL_PRICE_LOW_m_DAY,
        SYMBOL_PRICE_HIGH_m_DAY,
        SYMBOL_PRICE_MA_m_DAY,
        SYMBOL_ATR_m_DAY,
        SYMBOL_HL_ATR_RATIO_m_DAY,
        # indicators
        SYMBOL_VAL_DAY_CHANGE,
        SYMBOL_VAL_LEVEL_MINUS_x,
        SYMBOL_VAL_LEVEL_PLUS_x,
        SYMBOL_TOTAL_NUMBER,
        SYMBOL_NUMBER_PER_LEVEL,
    ]

    def __init__(
        self, max_leg: int, control_df: pd.DataFrame, study_df: pd.DataFrame, msg_queue, dataport: DataPort,
        ma_days=(1,2,5,10,20), max_scalping_level: int=4, ma_day_level: int=2, logger=None):
        self._logger = logger if logger is not None else get_logger('market_analyzer')
        self._msg_queue = msg_queue  # market message queue
        self.ctrl_df = control_df
        self.study_df = study_df
        self._dataport = dataport
        self.max_leg = max_leg
        self.ma_days = ma_days
        self.max_scalping_level = max_scalping_level
        self.ma_day_level = ma_day_level

        # Add extra study_df columns
        if self.max_leg > 1:
            self._studies = {
                x.format(i=leg+1, m=m, x=l+1)
                for leg in range(self.max_leg) for m in self.ma_days for l in range(self.max_scalping_level)
                for x in self.PAIR_STUDIES}
        else:
            self._studies = {
                x.format(m=m, x=l+1)
                for m in self.ma_days for l in range(self.max_scalping_level)
                for x in self.SINGLE_STUDIES}
        for col in self._studies:
            if col not in self.study_df:
                self.study_df[col] = [np.nan] * len(self.study_df)

        # Pair and symbol defs and streaming
        if self.max_leg > 1:
            self._watch_symbols = {
                s for i in range(1, self.max_leg+1)
                for s in self.ctrl_df[LEG_i_SYMBOL.format(i=i)] if s is not None}
        else:
            self._watch_symbols = {s for s in self.ctrl_df[SYMBOL] if s is not None}
        self._logger.debug(self._watch_symbols)
        self._price_decimal_lookup = {}  # number of decimals in price
        if self.max_leg > 1:
            self._symbol_pair_lookup = defaultdict(list)
            for p_name, p in self.ctrl_df.iterrows():
                if p_name is None:
                    continue
                for i in range(1, self.max_leg+1):
                    leg_symbol_col = LEG_i_SYMBOL.format(i=i)
                    self._symbol_pair_lookup[p[leg_symbol_col].upper()].append((p_name, i))
            self._logger.debug(self._symbol_pair_lookup)

        # Historical data objects
        self._specs_df = None
        self._symbol_to_id = {}
        self._ticker_to_symbol = {}
        self._ticker_to_id = {}
        self._id_to_ticker = {}
        self._id_to_symbol = {}
        self._watch_ids = {}
        self._margin_comm_ratio_df = None

        # Status and controls
        self._update_study_on_tick_thread = None
        self._update_study_on_tick_thread_running = False
        self._update_study_on_knobs_thread = None
        self._update_study_on_knobs_thread_running = False

    @staticmethod
    def shift_datatime_index_for_night_open(data_df: pd.DataFrame):
        data_df.index = data_df.index.shift(12, freq='H')  # shift UTC 13-7 to 1-19 on close day
        idx_sm = data_df.index.to_series()
        idx_sm = idx_sm.transform(lambda x: x+timedelta(days=2) if x.weekday()==5 else x)  # Merge Friday night market to Monday        
        data_df.index = pd.DatetimeIndex(idx_sm)

    def update_specs(self):
        self._logger.info("market_analyzer.update_specs ...")
        (self._specs_df, self._symbol_to_id, self._ticker_to_symbol, self._ticker_to_id, self._id_to_ticker,
        self._id_to_symbol) = self._dataport.query_specs(self._watch_symbols)
        self._watch_ids = {self._symbol_to_id[symbol] for symbol in self._watch_symbols}
        for i in range(1, self.max_leg+1):
            symbol_col = [SYMBOL, LEG_i_SYMBOL.format(i=i)][int(self.max_leg>1)]
            ticker_delta_col = [TICKER_DELTA, LEG_i_TICKER_DELTA.format(i=i)][int(self.max_leg>1)]
            high_limit_col = [HIGH_LIMIT, LEG_i_HIGH_LIMIT.format(i=i)][int(self.max_leg>1)]
            low_limit_col = [LOW_LIMIT, LEG_i_LOW_LIMIT.format(i=i)][int(self.max_leg>1)]
            settle_col = [SETTLE, LEG_i_SETTLE.format(i=i)][int(self.max_leg>1)]
            leg_delta, leg_high_limit, leg_low_limit, leg_settle = [], [], [], []
            for symbol in self.ctrl_df[symbol_col]:
                if symbol is None:
                    delta, high_limit, low_limit, settle = None, None, None, None
                else:
                    delta, high_limit, low_limit, settle = self._specs_df.loc[
                        self._symbol_to_id[symbol]][[TICKER_DELTA, HIGH_LIMIT, LOW_LIMIT, SETTLE]]
                leg_delta.append(delta)
                leg_high_limit.append(high_limit)
                leg_low_limit.append(low_limit)
                leg_settle.append(settle)
            self.ctrl_df[ticker_delta_col] = leg_delta
            self.ctrl_df[high_limit_col] = leg_high_limit
            self.ctrl_df[low_limit_col] = leg_low_limit
            self.ctrl_df[settle_col] = leg_settle

    def update_margin_comm_ratio(self, account_id):
        self._logger.info("market_analyzer.update_margin_comm_ratio ...")
        self._margin_comm_ratio_df = self._dataport.query_margin_comm_ratio(account_id, self._watch_ids)
        for i in range(1, self.max_leg+1):
            symbol_col = [SYMBOL, LEG_i_SYMBOL.format(i=i)][int(self.max_leg>1)]
            margin_ratio_long_col = [MARGIN_RATIO_LONG, LEG_i_MARGIN_RATIO_LONG.format(i=i)][int(self.max_leg>1)]
            margin_ratio_short_col = [MARGIN_RATIO_SHORT, LEG_i_MARGIN_RATIO_SHORT.format(i=i)][int(self.max_leg>1)]
            comm_ratio_type_col = [COMM_RATIO_TYPE, LEG_i_COMM_RATIO_TYPE.format(i=i)][int(self.max_leg>1)]
            comm_ratio_open_col = [COMM_OPEN_RATIO, LEG_i_COMM_RATIO_OPEN.format(i=i)][int(self.max_leg>1)]
            comm_ratio_close_col = [COMM_CLOSE_RATIO, LEG_i_COMM_RATIO_CLOSE.format(i=i)][int(self.max_leg>1)]
            comm_ratio_close_today_col = [
                COMM_CLOSE_TODAY_RATIO, LEG_i_COMM_RATIO_CLOSE_TODAY.format(i=i)][int(self.max_leg>1)]

            leg_margin_ratio_long, leg_margin_ratio_short = [], []
            leg_comm_ratio_type, leg_comm_open, leg_comm_close, leg_comm_close_today = [], [], [], []
            for symbol in self.ctrl_df[symbol_col]:
                if symbol is None:
                    margin_ratio_long, margin_ratio_short = None, None
                    comm_ratio_type, comm_open, comm_close, comm_close_today = None, None, None, None
                else:
                    (margin_ratio_long, margin_ratio_short, comm_ratio_type, comm_open, comm_close, comm_close_today
                    ) = self._margin_comm_ratio_df.loc[self._symbol_to_id[symbol]][[
                        MARGIN_RATIO_LONG, MARGIN_RATIO_SHORT, COMM_RATIO_TYPE, COMM_OPEN_RATIO, COMM_CLOSE_RATIO,
                        COMM_CLOSE_TODAY_RATIO]]
                leg_margin_ratio_long.append(margin_ratio_long)
                leg_margin_ratio_short.append(margin_ratio_short)
                leg_comm_ratio_type.append(comm_ratio_type)
                leg_comm_open.append(comm_open)
                leg_comm_close.append(comm_close)
                leg_comm_close_today.append(comm_close_today)
            self.ctrl_df[margin_ratio_long_col] = leg_margin_ratio_long
            self.ctrl_df[margin_ratio_short_col] = leg_margin_ratio_short
            self.ctrl_df[comm_ratio_type_col] = leg_comm_ratio_type
            self.ctrl_df[comm_ratio_open_col] = leg_comm_open
            self.ctrl_df[comm_ratio_close_col] = leg_comm_close
            self.ctrl_df[comm_ratio_close_today_col] = leg_comm_close_today

    def _update_one_pair_stats(self, pair_name, symbol_dfs_m):
        """
        Compute statistics of one pair and update study_df.
        Note: Support 2-leg pair now.
        """
        pair_ctrl = self.ctrl_df.loc[pair_name]
        pair_symbols = [pair_ctrl[LEG_i_SYMBOL.format(i=i)] for i in range(1, self.max_leg+1)]
        pair_delta = [pair_ctrl[LEG_i_TICKER_DELTA.format(i=i)] for i in range(1, self.max_leg+1)]
        pair_N = [pair_ctrl[LEG_i_N.format(i=i)] for i in range(1, self.max_leg+1)]

        # Select pair 1m data
        leg_high_m = [symbol_dfs_m[s][HIGH_PRICE] for s in pair_symbols]
        leg_low_m = [symbol_dfs_m[s][LOW_PRICE] for s in pair_symbols]
        leg_mid_m = [symbol_dfs_m[s][[BID_PRICE, ASK_PRICE]].mean(axis=1) for s in pair_symbols]

        # Compute 1m pair close value
        leg_val_m = [leg_mid_m[leg] * pair_delta[leg] * pair_N[leg] for leg in range(self.max_leg)]
        if self.max_leg == 2:
            pair_close_m = leg_val_m[0] - leg_val_m[1]
        elif self.max_leg == 3:
            pair_close_m = leg_val_m[0] + leg_val_m[2] - 2 * leg_val_m[1]

        # Build 1m df for different back date windows
        df_m_dict = {}
        for i in range(self.max_leg):
            df_m_dict.update({
                'leg{}_high'.format(i+1): leg_high_m[i],
                'leg{}_low'.format(i+1): leg_low_m[i],
                'leg{}_close'.format(i+1): leg_mid_m[i],
            })
        df_m_dict.update({'pair_close': pair_close_m})
        df_m = pd.DataFrame.from_dict(df_m_dict)
        trading_dates = pd.DatetimeIndex(sorted(set(df_m.index.date)))
        df_m_backdates = [
            df_m.loc[trading_dates[-1*n_days].date():] if n_days <= len(trading_dates)
            else pd.DataFrame(columns=df_m.columns)
            for n_days in self.ma_days]

        # Compute pair high/low/MA/corr for different back date windows
        pair_high = [df['pair_close'].max() for df in df_m_backdates]
        pair_low = [df['pair_close'].min() for df in df_m_backdates]
        pair_ma = [df['pair_close'].mean() for df in df_m_backdates]
        # ----------- Only 2 legs starting from here ---------------
        pair_corr = [
            np.nan if df.empty else
            df[['leg1_close', 'leg2_close']].pct_change()[1:].corr().values[0][1]
            for df in df_m_backdates]

        # ATR of legs and pair
        leg1_high_d = df_m['leg1_high'].groupby(df_m.index.date).max()
        leg1_low_d = df_m['leg1_low'].groupby(df_m.index.date).min()
        leg1_close_d = df_m['leg1_close'].groupby(df_m.index.date).tail(1)
        leg1_close_d.index = leg1_close_d.index.date
        leg1_prv_close_d = leg1_close_d.shift(1)
        
        leg2_high_d = df_m['leg2_high'].groupby(df_m.index.date).max()
        leg2_low_d = df_m['leg2_low'].groupby(df_m.index.date).min()
        leg2_close_d = df_m['leg2_close'].groupby(df_m.index.date).tail(1)
        leg2_close_d.index = leg2_close_d.index.date
        leg2_prv_close_d = leg2_close_d.shift(1)

        pair_high_d = df_m['pair_close'].groupby(df_m.index.date).max()
        pair_low_d = df_m['pair_close'].groupby(df_m.index.date).min()
        pair_close_d = df_m['pair_close'].groupby(df_m.index.date).tail(1)
        pair_close_d.index = pair_close_d.index.date
        pair_prv_close_d = pair_close_d.shift(1)

        df_atr = pd.DataFrame.from_dict({
            'leg1_high_d': leg1_high_d,
            'leg1_low_d': leg1_low_d,
            'leg1_prv_close_d': leg1_prv_close_d,            
            'leg2_high_d': leg2_high_d,
            'leg2_low_d': leg2_low_d,
            'leg2_prv_close_d': leg2_prv_close_d,
            'pair_high_d': pair_high_d,
            'pair_low_d': pair_low_d,
            'pair_prv_close_d': pair_prv_close_d,
        })
        
        tr1 = (df_atr['leg1_high_d'] - df_atr['leg1_low_d']).abs()
        tr2 = (df_atr['leg1_high_d'] - df_atr['leg1_prv_close_d']).abs()
        tr3 = (df_atr['leg1_low_d'] - df_atr['leg1_prv_close_d']).abs()
        leg1_true_range_d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        leg1_atr = [
            leg1_true_range_d[-n_days:].mean() if n_days <= len(leg1_true_range_d) else np.nan
            for n_days in self.ma_days]
        
        tr1 = (df_atr['leg2_high_d'] - df_atr['leg2_low_d']).abs()
        tr2 = (df_atr['leg2_high_d'] - df_atr['leg2_prv_close_d']).abs()
        tr3 = (df_atr['leg2_low_d'] - df_atr['leg2_prv_close_d']).abs()
        leg2_true_range_d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        leg2_atr = [
            leg2_true_range_d[-n_days:].mean() if n_days <= len(leg2_true_range_d) else np.nan
            for n_days in self.ma_days]

        leg_atr = [leg1_atr, leg2_atr]

        tr1 = (df_atr['pair_high_d'] - df_atr['pair_low_d']).abs()
        tr2 = (df_atr['pair_high_d'] - df_atr['pair_prv_close_d']).abs()
        tr3 = (df_atr['pair_low_d'] - df_atr['pair_prv_close_d']).abs()
        pair_true_range_d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        pair_atr = [
            pair_true_range_d[-n_days:].mean() if n_days <= len(pair_true_range_d) else np.nan
            for n_days in self.ma_days]
        
        # Find pair close value of last day
        pair_prev_close = pair_close_m[-1]
        leg_prev_close = [leg1_close_d[-1], leg2_close_d[-1]]

        # Daily value changes and ratios
        leg_daily_change = [[atr * pair_delta[leg] for atr in leg_atr[leg]] for leg in range(self.max_leg)]
        leg_daily_change_N = [[ch * pair_N[leg] for ch in leg_daily_change[leg]] for leg in range(self.max_leg)]
        pair_ratio =[ch1 / ch2 for (ch1, ch2) in zip(*leg_daily_change)]
        # ----------- Only 2 legs ending here -------------

        # Max/Min risk distances
        max_risk_distance = [max(ch_m) for ch_m in zip(*leg_daily_change_N)]
        min_risk_distance = [max_r * (1 - rho) for (max_r, rho) in zip(max_risk_distance, pair_corr)]

        # (H-L) / PairATR
        hl_pair_atr_ratio = [round((h-l)/p_atr, 1) for h, l, p_atr in zip(pair_high, pair_low, pair_atr)]

        # (H-L) / MinRiskDistance
        hl_min_distance_ratio = [round((h-l)/min_r, 1) for h, l, min_r in zip(pair_high, pair_low, min_risk_distance)]

        # Update stats to study_df
        self.study_df.at[pair_name, PAIR_VAL_PREV_CLOSE] = pair_prev_close
        for leg in range(self.max_leg):
            self.study_df.at[pair_name, LEG_i_PREV_CLOSE.format(i=leg+1)] = leg_prev_close[leg]
        for k, v in {
            PAIR_VAL_HIGH_m_DAY: pair_high,
            PAIR_VAL_LOW_m_DAY: pair_low,
            PAIR_VAL_MA_m_DAY: pair_ma,
            PAIR_CORR_m_DAY: pair_corr,
            PAIR_ATR_m_DAY: pair_atr,
            PAIR_DAY_CHANGE_RATIO_m_DAY: pair_ratio,
            PAIR_VAL_LEVEL_MIN_DIST_m_DAY: min_risk_distance,
            PAIR_VAL_LEVEL_MAX_DIST_m_DAY: max_risk_distance,
            PAIR_HL_ATR_RATIO_m_DAY: hl_pair_atr_ratio,
            PAIR_HL_MIN_RISK_RATIO_m_DAY: hl_min_distance_ratio,
            }.items():
            for ma_day_idx in range(len(self.ma_days)):
                m = self.ma_days[ma_day_idx]
                self.study_df.at[pair_name, k.format(m=m)] = v[ma_day_idx]
        for k, v in {
            LEG_i_ATR_m_DAY: [leg1_atr, leg2_atr],
            LEG_i_DAY_CHANGE_m_DAY: leg_daily_change,
            LEG_i_DAY_CHANGE_N_m_DAY: leg_daily_change_N,
            }.items():
            for leg in range(self.max_leg):
                for ma_day_idx in range(len(self.ma_days)):
                    m = self.ma_days[ma_day_idx]
                    self.study_df.at[pair_name, k.format(i=leg+1, m=m)] = v[leg][ma_day_idx]

    def update_pair_stats(self):
        self._logger.info("market_analyzer.update_pair_stats ...")

        # Query historical data
        self._logger.info("market_analyzer.update_pair_stats: Query 1 month data...")
        df_m, trading_dates = self._dataport.query_1month_data(self._watch_ids, self._dataport.DB_HIST_1M)
        symbol_dfs_m = self._dataport.split_data_per_symbol(df_m, self._id_to_symbol)
        for df_m in symbol_dfs_m.values():
            MarketAnalyzer.shift_datatime_index_for_night_open(df_m)

        # Compute stats for each pair and update to study_df
        self._logger.info("market_analyzer.update_pair_stats: Computing ...")
        for pair_name in self.ctrl_df.index:
            if pair_name is None:
                continue
            self._update_one_pair_stats(pair_name, symbol_dfs_m)

    def _update_one_single_stats(self, symbol, symbol_dfs_m, symbol_dfs_d):
        """
        Compute statistics of a single contract and update study_df.
        """
        symbol_ctrl = self.ctrl_df.loc[symbol]
        symbol_delta = symbol_ctrl[TICKER_DELTA]
        max_margin_ratio = symbol_ctrl[[MARGIN_RATIO_LONG, MARGIN_RATIO_SHORT]].max()

        # Select 1m data
        symbol_high_m = symbol_dfs_m[symbol][HIGH_PRICE]
        symbol_low_m = symbol_dfs_m[symbol][LOW_PRICE]
        symbol_mid_m = symbol_dfs_m[symbol][[BID_PRICE, ASK_PRICE]].mean(axis=1)

        # Compute 1m close value
        symbol_val_m = symbol_mid_m * symbol_delta

        # Build 1m df for different back date windows
        df_m_dict = {
            'high': symbol_high_m,
            'low': symbol_low_m,
            'close': symbol_mid_m,
            'value': symbol_val_m,
        }
        df_m = pd.DataFrame.from_dict(df_m_dict)
        trading_dates = pd.DatetimeIndex(sorted(set(df_m.index.date)))
        df_m_backdates = [
            df_m.loc[trading_dates[-1*n_days].date():] if n_days <= len(trading_dates)
            else pd.DataFrame(columns=df_m.columns)
            for n_days in self.ma_days]

        # Compute pair high/low/MA/corr for different back date windows
        symbol_high = [df['high'].max() for df in df_m_backdates]
        symbol_low = [df['low'].min() for df in df_m_backdates]
        symbol_ma = [df['close'].mean() for df in df_m_backdates]            

        # ATR
        symbol_high_d = df_m['high'].groupby(df_m.index.date).max()
        symbol_low_d = df_m['low'].groupby(df_m.index.date).min()
        symbol_close_d = df_m['close'].groupby(df_m.index.date).tail(1)
        symbol_close_d.index = symbol_close_d.index.date
        symbol_prv_close_d = symbol_close_d.shift(1)
        df_atr = pd.DataFrame.from_dict({
            'symbol_high_d': symbol_high_d,
            'symbol_low_d': symbol_low_d,
            'symbol_prv_close_d': symbol_prv_close_d,            
        })        
        tr1 = (df_atr['symbol_high_d'] - df_atr['symbol_low_d']).abs()
        tr2 = (df_atr['symbol_high_d'] - df_atr['symbol_prv_close_d']).abs()
        tr3 = (df_atr['symbol_low_d'] - df_atr['symbol_prv_close_d']).abs()
        symbol_true_range_d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        symbol_atr = [
            symbol_true_range_d[-n_days:].mean() if n_days <= len(symbol_true_range_d) else np.nan
            for n_days in self.ma_days]
        
        # Find close price, value, volume, open interest of last day
        symbol_prev_close = symbol_close_d[-1]
        symbol_margin = max_margin_ratio * symbol_delta * symbol_prev_close
        symbol_prev_val = symbol_val_m[-1]
        symbol_prev_open_interest = self._specs_df.at[self._symbol_to_id[symbol], OPEN_INTEREST]
        symbol_prev_open_interest_capital = symbol_prev_open_interest * symbol_prev_close * symbol_delta * max_margin_ratio

        # Daily changes
        symbol_day_change = [atr * symbol_delta for atr in symbol_atr]

        # (H-L) / ATR
        symbol_hl_atr_ratio = [round((h-l)/p_atr, 1) for h, l, p_atr in zip(symbol_high, symbol_low, symbol_atr)]

        # Get last day volume from daily data
        symbol_prev_volume = symbol_dfs_d[symbol][VOLUME][-1]

        # Update stats to study_df
        for k, v in {
            MARGIN_VAL: symbol_margin,
            PREV_CLOSE: symbol_prev_close,
            SYMBOL_PREV_VAL: symbol_prev_val,
            PREV_VOLUME: symbol_prev_volume,
            PREV_OPEN_INTEREST: symbol_prev_open_interest,
            PREV_OPEN_INTEREST_CAPITAL: symbol_prev_open_interest_capital,
        }.items():
            self.study_df.at[symbol, k] = v
        for k, v in {
            SYMBOL_VAL_DAY_CHANGE_m_DAY: symbol_day_change,            
            SYMBOL_PRICE_LOW_m_DAY: symbol_low,
            SYMBOL_PRICE_HIGH_m_DAY: symbol_high,
            SYMBOL_PRICE_MA_m_DAY: symbol_ma,
            SYMBOL_ATR_m_DAY: symbol_atr,
            SYMBOL_HL_ATR_RATIO_m_DAY: symbol_hl_atr_ratio,
        }.items():
            for ma_day_idx in range(len(self.ma_days)):
                m = self.ma_days[ma_day_idx]
                self.study_df.at[symbol, k.format(m=m)] = v[ma_day_idx]

    def update_single_stats(self):
        self._logger.info("market_analyzer.update_single_stats ...")

        # Query historical data
        self._logger.info("market_analyzer.update_single_stats: Query 1 month data...")
        df_m, trading_dates = self._dataport.query_1month_data(self._watch_ids, self._dataport.DB_HIST_1M)
        symbol_dfs_m = self._dataport.split_data_per_symbol(df_m, self._id_to_symbol)
        for df_m in symbol_dfs_m.values():
            MarketAnalyzer.shift_datatime_index_for_night_open(df_m)
        df_d = self._dataport.query_hist_data(
            self._dataport.DB_HIST_1D, symbol_ids=self._watch_ids, start=trading_dates[-5])
        symbol_dfs_d = self._dataport.split_data_per_symbol(df_d, self._id_to_symbol)
        # Compute stats for each symbol and update to study_df
        self._logger.info("market_analyzer.update_single_stats: Computing ...")
        for symbol in self.ctrl_df.index:
            if symbol is None:
                continue
            self._update_one_single_stats(symbol, symbol_dfs_m, symbol_dfs_d)

    def _update_one_pair_indicators(self, pair_name):
        """
        Compute indicators of a single pair based on statistics.
        """
        # Retrieve stats
        leg_day_change = [
            self.study_df.at[pair_name, LEG_i_DAY_CHANGE_m_DAY.format(i=leg+1, m=self.ma_day_level)]
            for leg in range(self.max_leg)
        ]
        leg_day_change_N = [
            self.study_df.at[pair_name, LEG_i_DAY_CHANGE_N_m_DAY.format(i=leg+1, m=self.ma_day_level)]
            for leg in range(self.max_leg)
        ]
        pair_var = self.ctrl_df.at[pair_name, PAIR_VAR]
        pair_prev_close, min_risk_distance_orig, max_risk_distance = self.study_df.loc[pair_name][[
            PAIR_VAL_PREV_CLOSE,
            PAIR_VAL_LEVEL_MIN_DIST_m_DAY.format(m=self.ma_day_level),
            PAIR_VAL_LEVEL_MAX_DIST_m_DAY.format(m=self.ma_day_level),]]
        pair_prev_close_offset, min_risk_distance_scale = self.ctrl_df.loc[pair_name][[
            PAIR_VAL_PREV_CLOSE_OFFSET,
            PAIR_VAL_LEVEL_MIN_DIST_SCALE]]
        pair_prev_close += pair_prev_close_offset
        min_risk_distance = min_risk_distance_orig * min_risk_distance_scale
        
        # Compute trading indicators
        level_1 = [pair_prev_close + sign * 0.5 * min_risk_distance for sign in (-1, 1)]
        level_2 = [pair_prev_close + sign * min_risk_distance for sign in (-1, 1)]
        level_3 = [pair_prev_close + sign * min(1.5 * min_risk_distance, max_risk_distance) for sign in (-1, 1)]
        level_4 = [pair_prev_close + sign * max(2 * min_risk_distance, max_risk_distance) for sign in (-1, 1)]
        level_minus, level_plus = [l for l in zip(level_1, level_2, level_3, level_4)]
        max_pair_num = round(pair_var / min_risk_distance, 1)
        pair_num_per_level = round(max_pair_num / self.max_scalping_level, 1)

        # Update indicators to study_df
        for k, v in {
            PAIR_VAL_LEVEL_MAX_DIST: max_risk_distance,
            PAIR_VAL_LEVEL_MIN_DIST: min_risk_distance_orig,
            PAIR_TOTAL_NUMBER: max_pair_num,
            PAIR_NUMBER_PER_LEVEL: pair_num_per_level,
            }.items():
            self.study_df.at[pair_name, k] = v
        for k, v in {
            LEG_i_DAY_CHANGE: leg_day_change,
            LEG_i_DAY_CHANGE_N: leg_day_change_N,
            }.items():
            for leg in range(self.max_leg):
                self.study_df.at[pair_name, k.format(i=leg+1)] = v[leg]
        for k, v in {
            PAIR_VAL_LEVEL_MINUS_x: level_minus,
            PAIR_VAL_LEVEL_PLUS_x: level_plus,
            }.items():
            for l in range(self.max_scalping_level):
                self.study_df.at[pair_name, k.format(x=l+1)] = v[l]

    def update_pair_indicators(self):
        self._logger.info("market_analyzer.update_pair_indicators: Computing ...")
        for pair_name in self.ctrl_df.index:
            if pair_name is None:
                continue
            self._update_one_pair_indicators(pair_name)        

    def update_pair_margin_value(self):
        leg1_margin_val = (self.ctrl_df[LEG_i_N.format(i=1)] * self.ctrl_df[LEG_i_TICKER_DELTA.format(i=1)]
         * self.study_df[LEG_i_PREV_CLOSE.format(i=1)] * self.ctrl_df[LEG_i_MARGIN_RATIO_LONG.format(i=1)])
        leg2_margin_val = (self.ctrl_df[LEG_i_N.format(i=2)] * self.ctrl_df[LEG_i_TICKER_DELTA.format(i=2)]
         * self.study_df[LEG_i_PREV_CLOSE.format(i=2)] * self.ctrl_df[LEG_i_MARGIN_RATIO_SHORT.format(i=2)])
        self.study_df[PAIR_TOTAL_MARGIN] = (leg1_margin_val + leg2_margin_val).round(2)

    def _update_one_single_indicators(self, symbol):
        """
        Compute indicators of one symbol based on statistics.
        """
        # Retrieve stats
        symbol_var = self.ctrl_df.at[symbol, SYMBOL_VAR]
        day_change = self.study_df.at[symbol, SYMBOL_VAL_DAY_CHANGE_m_DAY.format(m=self.ma_day_level)]
        prev_close = self.study_df.at[symbol, PREV_CLOSE]
        symbol_atr = self.study_df.at[symbol, SYMBOL_ATR_m_DAY.format(m=self.ma_day_level)]
        
        # Compute trading indicators
        level_minus = [(prev_close - a * symbol_atr) for a in (0.5, 1, 1.5, 2.0)]
        level_plus = [(prev_close + a * symbol_atr) for a in (0.5, 1, 1.5, 2.0)]
        max_num = round(symbol_var / day_change, 1)
        num_per_level = round(max_num / self.max_scalping_level, 1)

        # Update indicators to study_df
        for k, v in {
            SYMBOL_VAL_DAY_CHANGE: day_change,
            SYMBOL_TOTAL_NUMBER: max_num,
            SYMBOL_NUMBER_PER_LEVEL: num_per_level,
            }.items():
            self.study_df.at[symbol, k] = v
        for k, v in {
            SYMBOL_VAL_LEVEL_MINUS_x: level_minus,
            SYMBOL_VAL_LEVEL_PLUS_x: level_plus,
            }.items():
            for l in range(self.max_scalping_level):
                self.study_df.at[symbol, k.format(x=l+1)] = v[l]

    def update_single_indicators(self):
        self._logger.info("market_analyzer.update_single_indicators: Computing ...")
        for symbol in self.ctrl_df.index:
            if symbol is None:
                continue
            self._update_one_single_indicators(symbol)

    def update_daily_studies(self, account_id):
        self.update_margin_comm_ratio(account_id)
        if self.max_leg > 1:
            self.update_pair_stats()
            self.update_pair_indicators()
            self.update_pair_margin_value()
        else:
            self.update_single_stats()
            self.update_single_indicators()

    def _update_pair_study_on_tick(self, tick_msg):
        symbol = tick_msg[SYMBOL].upper()
        if symbol not in self._watch_symbols:
            return

        bid, ask = tick_msg[BID_PRICE], tick_msg[ASK_PRICE]
        try:
            decimal_number = self._price_decimal_lookup[symbol]
        except KeyError:
            decimal_number = get_number_of_decimal(tick_msg[TICK_SIZE])
            self._price_decimal_lookup[symbol] = decimal_number
        mid = round((bid + ask) / 2.0, decimal_number)

        for p_name, i in self._symbol_pair_lookup[symbol]:
            # update self leg price
            self.study_df.at[p_name, LEG_i_PRICE.format(i=i)] = mid
            
            # update pair value
            p_ctrl, p_study = self.ctrl_df.loc[p_name], self.study_df.loc[p_name]
            try:
                if self.max_leg == 2:
                    pair_value = (
                        p_ctrl[LEG_i_N.format(i=1)] * p_ctrl[LEG_i_TICKER_DELTA.format(i=1)] * p_study[LEG_i_PRICE.format(i=1)]
                        - p_ctrl[LEG_i_N.format(i=2)] * p_ctrl[LEG_i_TICKER_DELTA.format(i=2)] * p_study[LEG_i_PRICE.format(i=2)])
                elif self.max_leg == 3:  # butterfly
                    pair_value = (
                        p_ctrl[LEG_i_N.format(i=1)] * p_ctrl[LEG_i_TICKER_DELTA.format(i=1)] * p_study[LEG_i_PRICE.format(i=1)]
                        + p_ctrl[LEG_i_N.format(i=3)] * p_ctrl[LEG_i_TICKER_DELTA.format(i=3)] * p_study[LEG_i_PRICE.format(i=3)]
                        - 2 * p_ctrl[LEG_i_N.format(i=2)] * p_ctrl[LEG_i_TICKER_DELTA.format(i=2)] * p_study[LEG_i_PRICE.format(i=2)])
            except TypeError:  # One of fields is None, NaN
                continue
            self.study_df.at[p_name, PAIR_VAL] = pair_value

            # update pair value day high/low
            try:
                if pair_value > p_study[PAIR_VAL_DAY_HIGH]:
                    self.study_df.at[p_name, PAIR_VAL_DAY_HIGH] = pair_value
            except TypeError:  # PAIR_VAL_DAY_HIGH is None, NaN
                self.study_df.at[p_name, PAIR_VAL_DAY_HIGH] = pair_value                
            try:
                if pair_value < p_study[PAIR_VAL_DAY_LOW]:
                    self.study_df.at[p_name, PAIR_VAL_DAY_LOW] = pair_value
            except TypeError:  # PAIR_VAL_DAY_LOW is None, NaN
                self.study_df.at[p_name, PAIR_VAL_DAY_LOW] = pair_value                

    def _update_single_study_on_tick(self, tick_msg):
        symbol = tick_msg[SYMBOL].upper()
        if symbol not in self._watch_symbols:
            return

        bid, ask, high, low = tick_msg[BID_PRICE], tick_msg[ASK_PRICE], tick_msg[HIGH_PRICE], tick_msg[LOW_PRICE]
        try:
            decimal_number = self._price_decimal_lookup[symbol]
        except KeyError:
            decimal_number = get_number_of_decimal(tick_msg[TICK_SIZE])
            self._price_decimal_lookup[symbol] = decimal_number
        mid = round((bid + ask) / 2.0, decimal_number)
        self.study_df.at[symbol, MID_PRICE] = mid
        self.study_df.at[symbol, MID_PRICE_DAY_HIGH] = high
        self.study_df.at[symbol, MID_PRICE_DAY_LOW] = low
        open_interest_change = tick_msg[OPEN_INTEREST] - self.study_df.at[symbol, PREV_OPEN_INTEREST]
        self.study_df.at[symbol, OPEN_INTEREST_TICK_CHANGE] = open_interest_change

    def _update_study_on_tick_engine(self):
        update_func = [self._update_single_study_on_tick, self._update_pair_study_on_tick][self.max_leg>1]
        while self._update_study_on_tick_thread_running:
            try:
                msg = self._msg_queue.get(block=True, timeout=.5)
                update_func(msg)
            except Empty as e:
                pass
            except BaseException as e:
                self._logger.error(
                    'market_analyzer._update_study_on_tick_engine error: {}'.format(e.__repr__()), exc_info=True)

    def start_update_study_on_tick(self):
        self._update_study_on_tick_thread = Thread(target=self._update_study_on_tick_engine)
        self._update_study_on_tick_thread_running = True
        self._update_study_on_tick_thread.start()
        self._logger.info("market_analyzer._update_study_on_tick_thread started.")

    def stop_update_study_on_tick(self):
        if self._update_study_on_tick_thread is None or not self._update_study_on_tick_thread.isAlive():
            return
        self._update_study_on_tick_thread_running = False
        if self._update_study_on_tick_thread.isAlive():
            self._update_study_on_tick_thread.join()
        self._logger.info("market_analyzer._update_study_on_tick_thread stopped.")

    def _update_study_on_knobs_engine(self, time_interval=1.0):
        knobs = [PAIR_VAL_PREV_CLOSE_OFFSET, PAIR_VAL_LEVEL_MIN_DIST_SCALE]
        ctrl_df_last = self.ctrl_df.copy()
        while self._update_study_on_knobs_thread_running:
            try:
                ctrl_df_snapshot = self.ctrl_df.copy()
                cmp = (ctrl_df_last[knobs] == ctrl_df_snapshot[knobs]).all(axis=1)
                for pair_name in cmp[cmp==False].index:
                    self._update_one_pair_indicators(pair_name)
                ctrl_df_last = ctrl_df_snapshot
                sleep(time_interval)
            except Empty as e:
                pass
            except BaseException as e:
                self._logger.error(
                    'market_analyzer._update_study_on_knobs_engine error: {}'.format(e.__repr__()), exc_info=True)

    def start_update_study_on_knobs(self):
        if self.max_leg <= 1:
            return
        self._update_study_on_knobs_thread = Thread(target=self._update_study_on_knobs_engine)
        self._update_study_on_knobs_thread_running = True
        self._update_study_on_knobs_thread.start()
        self._logger.info("market_analyzer._update_study_on_knobs_thread started.")

    def stop_update_study_on_knobs(self):
        if self.max_leg <= 1:
            return
        if self._update_study_on_knobs_thread is None or not self._update_study_on_knobs_thread.isAlive():
            return
        self._update_study_on_knobs_thread_running = False
        if self._update_study_on_knobs_thread.isAlive():
            self._update_study_on_knobs_thread.join()
        self._logger.info("market_analyzer._update_study_on_knobs_thread stopped.")


# Unit test
def test():
    import os
    import sys
    from time import sleep
    from queue import Queue
    from ast import literal_eval
    from .excel_manager import ExcelManager
    from .dataport import DataPort
    from .china_globals import ALPHAMONGO_DB_DICT, ALPHAMONGO_DB_TICKTIME_FORMAT

    excel_manager = ExcelManager(
        max_leg=2, excel_file='DailyUpdate.xlsx', sheet_name='Pair', start_row=1,
        ma_days=(1,2,5,10,20), max_level=4)
    ctrl_df, study_df = excel_manager.load_spreadsheet()

    msg_queue = Queue()

    db_login = {'Host': '', 'Port': 3306, 'User': '', 'Password': '', 'Database': ''}
    data_port = DataPort(db_login, ALPHAMONGO_DB_DICT, ALPHAMONGO_DB_TICKTIME_FORMAT)

    analyzer = MarketAnalyzer(
        2, ctrl_df, study_df, msg_queue, data_port, ma_days=(1,2,5,10,20), ma_day_level=2)

    print("Test : Update specs, daily statistics. Press any key to continue.")
    _ = input()
    analyzer.update_specs()
    excel_manager.write_control()
    analyzer.update_daily_studies(account_id=67)
    excel_manager.write_study()
    print(ctrl_df)
    print(study_df)

    print("Test : Update study on tick. Press any key to continue.")
    _ = input()
    excel_manager.start_update_study()
    excel_manager.start_read_control()
    analyzer.start_update_study_on_tick()
    analyzer.start_update_study_on_knobs()
    while True:
        print("Input message dictionary. Press q to quit.\n")
        msg_str = input()
        if msg_str == 'q':
            break
        msg = literal_eval(msg_str)  # eval dict string 
        msg_queue.put(msg)
        sleep(0.001)

    analyzer.stop_update_study_on_knobs()
    analyzer.stop_update_study_on_tick()
    excel_manager.stop_read_control()
    excel_manager.stop_update_study()
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)    


if __name__ == '__main__':
    """
    {'SYMBOL': 'RB1910', 'BID_PRICE': 3738, 'ASK_PRICE': 3740, 'TICK_SIZE': 1}
    {'SYMBOL': 'HC1910', 'BID_PRICE': 3600, 'ASK_PRICE': 3602, 'TICK_SIZE': .1}
    {'SYMBOL': 'MA909', 'BID_PRICE': 302.25, 'ASK_PRICE': 308.75, 'TICK_SIZE': .25}
    """
    test()