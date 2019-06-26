# -*- coding: utf-8 -*-

"""
Read real-time market data, statistics, indicators and update Excel spreadsheet.
"""


import xlwings as xw
from threading import Thread, Lock
from time import sleep
import pandas as pd
from .utils import get_logger
from .constants import *


SINGLE_SHEET_COLUMN_DICT = {
    'Symbol': SYMBOL,
    'Ticker': TICKER,
    'Risk': SYMBOL_VAR,
    'Delta': TICKER_DELTA,
    'Price': MID_PRICE,
    'Price%': MID_PRICE_CHANGE_PERCENT_HL20,
    'DH': MID_PRICE_DAY_HIGH,
    'DL': MID_PRICE_DAY_LOW,
    'OIChg': OPEN_INTEREST_TICK_CHANGE,
    'OIChg%': OPEN_INTEREST_TICK_CHANGE_PERCENT,
    'Margin': MARGIN_RATIO_MAX,
    'PreCls': PREV_CLOSE,
    'PrcVal': SYMBOL_PREV_VAL,
    'PreVol': PREV_VOLUME,
    'PreOI': PREV_OPEN_INTEREST,
    'TotalCap': PREV_OPEN_INTEREST_CAPITAL,
    'Dchg': SYMBOL_VAL_DAY_CHANGE,
    'M{x}': SYMBOL_VAL_LEVEL_MINUS_x,
    'P{x}': SYMBOL_VAL_LEVEL_PLUS_x,
    'TP': SYMBOL_TOTAL_NUMBER,
    'P': SYMBOL_NUMBER_PER_LEVEL,
    'L{m}': SYMBOL_PRICE_LOW_m_DAY,
    'H{m}': SYMBOL_PRICE_HIGH_m_DAY,
    'MA{m}': SYMBOL_PRICE_MA_m_DAY,
    'A{m}': SYMBOL_ATR_m_DAY,
    '(H{m}-L{m})/A{m}': SYMBOL_HL_ATR_RATIO_m_DAY,
}
SINGLE_SHEET_CONTROL_COLUMNS = [TICKER, SYMBOL, SYMBOL_VAR, TICKER_DELTA]
SINGLE_SHEET_KNOB_COLUMNS = []
SINGLE_SHEET_STUDY_COLUMNS = [  # not used
    MID_PRICE, MID_PRICE_DAY_HIGH, MID_PRICE_DAY_LOW, OPEN_INTEREST_TICK_CHANGE, MARGIN_VAL, PREV_CLOSE,
    SYMBOL_PREV_VAL, PREV_VOLUME, PREV_OPEN_INTEREST, PREV_OPEN_INTEREST_CAPITAL, SYMBOL_VAL_DAY_CHANGE,
    SYMBOL_VAL_LEVEL_MINUS_x, SYMBOL_VAL_LEVEL_PLUS_x, SYMBOL_TOTAL_NUMBER, SYMBOL_NUMBER_PER_LEVEL,
    SYMBOL_PRICE_LOW_m_DAY, SYMBOL_PRICE_HIGH_m_DAY, SYMBOL_PRICE_MA_m_DAY,
    SYMBOL_ATR_m_DAY, SYMBOL_HL_ATR_RATIO_m_DAY,]

PAIR_SHEET_COLUMN_DICT = {
    'Name': PAIR_NAME,
    'CAP': PAIR_CAPITAL,
    'Risk': PAIR_VAR,
    'L{i}Sym': LEG_i_SYMBOL,
    'L{i}N': LEG_i_N,
    'L{i}D': LEG_i_TICKER_DELTA,
    'L{i}HLim': LEG_i_HIGH_LIMIT,
    'L{i}LLim': LEG_i_LOW_LIMIT,
    'L{i}Sett': LEG_i_SETTLE,
    'L{i}MgnL': LEG_i_MARGIN_RATIO_LONG,
    'L{i}MgnS': LEG_i_MARGIN_RATIO_SHORT,
    'L{i}CommT': LEG_i_COMM_RATIO_TYPE,
    'L{i}CommO': LEG_i_COMM_RATIO_OPEN, 
    'L{i}CommC': LEG_i_COMM_RATIO_CLOSE,
    'L{i}CommCT': LEG_i_COMM_RATIO_CLOSE_TODAY,
    'L{i}P': LEG_i_PRICE,
    'PairP': PAIR_VAL,
    'DH': PAIR_VAL_DAY_HIGH,
    'DL': PAIR_VAL_DAY_LOW,
    'L{i}NChg': LEG_i_DAY_CHANGE_N,
    'R{m}': PAIR_DAY_CHANGE_RATIO_m_DAY,
    'M{x}': PAIR_VAL_LEVEL_MINUS_x,
    'P{x}': PAIR_VAL_LEVEL_PLUS_x,
    'MinR': PAIR_VAL_LEVEL_MIN_DIST,
    'MinRScale': PAIR_VAL_LEVEL_MIN_DIST_SCALE,
    'MaxR': PAIR_VAL_LEVEL_MAX_DIST,
    'PreCls': PAIR_VAL_PREV_CLOSE,
    'PreClsOffset': PAIR_VAL_PREV_CLOSE_OFFSET,
    'Mgn': PAIR_TOTAL_MARGIN,
    'TP': PAIR_TOTAL_NUMBER,
    'P': PAIR_NUMBER_PER_LEVEL,
    'L{m}': PAIR_VAL_LOW_m_DAY,
    'H{m}': PAIR_VAL_HIGH_m_DAY,
    'MA{m}': PAIR_VAL_MA_m_DAY,
    'Cor{m}': PAIR_CORR_m_DAY,
    'L{i}A{m}': LEG_i_ATR_m_DAY,
    'PA{m}': PAIR_ATR_m_DAY,
    '(H{m}-L{m})/PA{m}': PAIR_HL_ATR_RATIO_m_DAY,
    '(H{m}-L{m})/minR{m}': PAIR_HL_MIN_RISK_RATIO_m_DAY,
}
PAIR_SHEET_CONTROL_COLUMNS = [
    PAIR_NAME, PAIR_CAPITAL, PAIR_VAR, LEG_i_SYMBOL, LEG_i_N, LEG_i_TICKER_DELTA, PAIR_VAL_LEVEL_MIN_DIST_SCALE,
    PAIR_VAL_PREV_CLOSE_OFFSET]
PAIR_SHEET_KNOB_COLUMNS = [PAIR_VAL_LEVEL_MIN_DIST_SCALE, PAIR_VAL_PREV_CLOSE_OFFSET]
PAIR_SHEET_STUDY_COLUMNS = []  # not used

def col_num_to_name(n):
    name = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        name = chr(65 + remainder) + name
    return name


class SpreadsheetNotFoundError(Exception):
    def __init__(self, spreadsheet_name):
        msg = "Spreadsheet '{}' does not exist.".format(spreadsheet_name)
        super(SpreadsheetNotFoundError, self).__init__(msg)


class ExcelManager:
    """
    Read real-time indicators and update Excel spreadsheet.
    Assumptions on spreadsheet schema:
    1. Only 1-line header.
    2. Column A contains pair names or symbol as index.
    3. Column B and thereafter contain only two kinds of columns: control, study (market, indicators and statistics).
    4. Control columns and study columns are continuous inside each block, i.e. they are not crossing each other.
    5. Control columns are before study columns.
    """

    def __init__(self, max_leg, excel_file, sheet_name, start_row, ma_days, max_level, logger=None):
        self._logger = logger if logger is not None else get_logger('excel_manager')
        try:
            self._wb = xw.Book(excel_file)
        except FileNotFoundError as e:
            raise e
        spreadsheet_names = [sht.name for sht in self._wb.sheets]
        if sheet_name not in spreadsheet_names:
            raise SpreadsheetNotFoundError(sheet_name)
        self._sht = self._wb.sheets[sheet_name]
        self.excel_file = excel_file
        self.sheet_name = sheet_name
        self._max_leg = max_leg
        self._ma_days = ma_days
        self._max_level = max_level

        # Excel spreadsheet schema
        self._start_row = start_row
        self._col_dict = {}
        for k, v in [SINGLE_SHEET_COLUMN_DICT, PAIR_SHEET_COLUMN_DICT][int(max_leg > 1)].items():
            for i in range(1, self._max_leg+1):
                for m in self._ma_days:
                    for x in range(1, self._max_level+1):
                        self._col_dict.update({k.format(i=i, m=m, x=x): v.format(i=i, m=m, x=x)})
        if max_leg > 1:  # pair control columns
            self._ctrl_cols = set()
            for c in PAIR_SHEET_CONTROL_COLUMNS:
                for i in range(1, self._max_leg+1):
                    self._ctrl_cols.add(c.format(i=i))
            self._ctrl_cols = list(self._ctrl_cols)
        else:  # single contract control columns
            self._ctrl_cols = set(SINGLE_SHEET_CONTROL_COLUMNS)
        self._knob_cols = [SINGLE_SHEET_KNOB_COLUMNS, PAIR_SHEET_KNOB_COLUMNS][int(max_leg > 1)]
        self._study_cols = None
        self._ctrl_range = None
        self._study_range = None
        last_cell = self._sht.used_range.last_cell
        self._last_row = last_cell.row
        self._last_col = col_num_to_name(last_cell.column)
        self._last_cell = '{}{}'.format(self._last_col, self._last_row)

        # DataFrames for reading control blocks and writing study blocks
        self.ctrl_df = None  # Update ctrl_df only in this Excel manager's updating thread.
        self.study_df = None  # Read-only to Excel manager's updating thread.

        # Threads
        self._read_control_thread = None
        self._read_control_thread_running = False
        self._update_study_thread = None
        self._update_study_thread_running = False

    def load_spreadsheet(self):
        df = self._sht.range('A{}:{}'.format(self._start_row, self._last_cell)).options(pd.DataFrame, index=False).value
        df.columns = [self._col_dict[c] if c in self._col_dict else c for c in df.columns]
        df = df.set_index([SYMBOL, PAIR_NAME][self._max_leg > 1], drop=False)
        ctrl_col_min_idx = df.columns.get_indexer_for(self._ctrl_cols).min()
        ctrl_col_max_idx = df.columns.get_indexer_for(self._ctrl_cols).max()
        self._ctrl_cols = list(df.columns[ctrl_col_min_idx : (ctrl_col_max_idx+1)])  # sorted control cols as in sheet
        self._study_cols = [c for c in df.columns if c not in self._ctrl_cols]  # data cols also in the order as sheet
        ctrl_start_cell = col_num_to_name(ctrl_col_min_idx+1) + str(self._start_row+1)
        ctrl_end_cell = col_num_to_name(ctrl_col_max_idx+1) + str(self._last_row)
        self._ctrl_range = '{}:{}'.format(ctrl_start_cell, ctrl_end_cell)
        study_start_cell = col_num_to_name(ctrl_col_max_idx+1+1) + str(self._start_row+1)
        self._study_range = '{}:{}'.format(study_start_cell, self._last_cell)
        self.ctrl_df, self.study_df = df[self._ctrl_cols], df[self._study_cols]
        self._logger.debug('Loaded spread sheet. ctrl_range: {} study_range: {} ctrl_cols: {} study_cols{}'.format(
            self._ctrl_range, self._study_range, self._ctrl_cols, self._study_cols))
        self._logger.debug('\ncontrol_df:\n{}\nstudy_df:\n{}\n'.format(self.ctrl_df, self.study_df))
        return self.ctrl_df, self.study_df

    def write_control(self):
        self._sht.range(self._ctrl_range).options(index=False, header=False).value = self.ctrl_df[self._ctrl_cols]

    def write_study(self):
        self._sht.range(self._study_range).options(index=False, header=False).value = self.study_df[self._study_cols]

    def _read_control_engine(self, time_interval=1.0):
        """
        Keep reading control blocks from spreadsheet to control dataframe at each time interval.
        """
        sht = xw.Book(self.excel_file).sheets[self.sheet_name]  # Must use thread's own xlwings object
        while self._read_control_thread_running:
            try:
                control_list = sht.range(self._ctrl_range).value
                df = pd.DataFrame(control_list, index=self.ctrl_df.index, columns=self._ctrl_cols)
                for c in self._knob_cols:
                    self.ctrl_df[c] = df[c]
                sleep(time_interval)
            except BaseException as e:
                self._logger.error('excel_manager._read_control_engine error: {}'.format(e.__repr__()), exc_info=True)

    def start_read_control(self):
        if self._max_leg <= 1:
            return
        self._read_control_thread = Thread(target=self._read_control_engine)
        self._read_control_thread_running = True
        self._read_control_thread.start()
        self._logger.info("excel_manager._read_control_thread started.")        

    def stop_read_control(self):
        if self._max_leg <= 1:
            return
        if self._read_control_thread is None or not self._read_control_thread.isAlive():
            return
        self._read_control_thread_running = False
        if self._read_control_thread.isAlive():
            self._read_control_thread.join()
        self._logger.info("excel_manager._read_control_thread stopped.")

    def _update_study_engine(self, time_interval=.4):
        """
        Keep writing study dataframe to spreadsheet at each time interval.
        """
        sht = xw.Book(self.excel_file).sheets[self.sheet_name]  # Must use thread's own xlwings object
        while self._update_study_thread_running:
            try:
                sht.range(self._study_range).options(index=False, header=False).value = self.study_df[self._study_cols]
                sleep(time_interval)
            except BaseException as e:
                self._logger.error('excel_manager._update_study_engine error: {}'.format(e.__repr__()), exc_info=True)

    def start_update_study(self):
        self._update_study_thread = Thread(target=self._update_study_engine)
        self._update_study_thread_running = True
        self._update_study_thread.start()
        self._logger.info("excel_manager._update_study_thread started.")        

    def stop_update_study(self):
        if self._update_study_thread is None or not self._update_study_thread.isAlive():
            return
        self._update_study_thread_running = False
        if self._update_study_thread.isAlive():
            self._update_study_thread.join()
        self._logger.info("excel_manager._update_study_thread stopped.")            


# Unit test
def test():
    import os, sys
    from ast import literal_eval

    excelmgr = ExcelManager(
        max_leg=2, excel_file='DailyUpdate.xlsx', sheet_name='Pair', start_row=1, ma_days=(1,2,5,10,20), max_level=4)
    ctrl_df, study_df = excelmgr.load_spreadsheet()
    print(excelmgr._ctrl_range)
    print(excelmgr._ctrl_cols)
    print(excelmgr._study_range)
    print(excelmgr._study_cols)
    print(ctrl_df)
    print(study_df)

    market_cols = [LEG_i_PRICE.format(i=1), LEG_i_PRICE.format(i=2), PAIR_VAL, PAIR_VAL_DAY_HIGH, PAIR_VAL_DAY_LOW]
    excelmgr.start_update_study()
    excelmgr.start_read_control()
    while True:
        print("Input [pair_name, leg1_price, leg2_price, pair_value, day_high, day_low]."
              "Press c to print ctrl_df. Press q to quit.\n")
        input_str = input()
        if input_str == 'c':
            print("{}\n".format(excelmgr.ctrl_df))
            continue
        if input_str == 'q':
            break
        input_list = literal_eval(input_str)  # eval list string 
        pair_name = input_list[0]
        excelmgr.study_df.at[pair_name, market_cols] = input_list[1:]
    
    excelmgr.stop_read_control()
    excelmgr.stop_update_study()
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)


if __name__ == '__main__':
    test()