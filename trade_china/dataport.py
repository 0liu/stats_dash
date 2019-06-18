# -*- coding: utf-8 -*-

"""
Historical database interface.
"""


import re
from typing import Iterable
import collections
from datetime import datetime, timedelta, date, time
import time as time_mod
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.sql import text
from .utils import get_logger, UTC, SHANGHAI
from .constants import *
from .china_globals import INVALID_DATES


class DataPort:
    """
    Interface object to database.
    """

    def __init__(self, db_login: dict, db_dict: dict, db_ticktime_format: dict, logger=None):
        """
        :db_login: 'host', 'port', 'user', 'password', 'database'.
        :db_dict: table and field names mapping.
        :db_ticktime_format: strptime()/strftime() format for historical and tick data.
        """
        self._logger = logger if logger is not None else get_logger('dataport')
        self._db_login = db_login
        self._db_dict = db_dict
        self._db_ticktime_format = db_ticktime_format
        self._mysql_engine, self._mysql_metadata = None, None
        self.create_mysql_engine()
        self.instruments_df = None  # instruments specs dataframe

        # constants mapping
        self.DB_SPECS = self._db_dict[DB_SPECS]
        self.DB_MARGIN_COMM = self._db_dict[DB_MARGIN_COMM]
        self.DB_HIST_1D = self._db_dict[DB_HIST_1D]
        self.DB_HIST_1H = self._db_dict[DB_HIST_1H]
        self.DB_HIST_30M = self._db_dict[DB_HIST_30M]
        self.DB_HIST_15M = self._db_dict[DB_HIST_15M]
        self.DB_HIST_5M = self._db_dict[DB_HIST_5M]
        self.DB_HIST_1M = self._db_dict[DB_HIST_1M]
        self.DB_HIST_1S = self._db_dict[DB_HIST_1S]
        self.DB_HIST_TICK = self._db_dict[DB_HIST_TICK]
        self.SYMBOL_ID = self._db_dict[SYMBOL_ID]
        self.SYMBOL = self._db_dict[SYMBOL]
        self.TICK_TIME = self._db_dict[TICK_TIME]
        self.VOLUME = self._db_dict[VOLUME]

    def create_mysql_engine(self):
        """
        Create engine for MySQL db.
        """
        db_conn = "mysql+pymysql://{user}:{password}@{host}:{port}/{database}".format(
            host=self._db_login['host'], port=self._db_login['port'], user=self._db_login['user'],
            password=self._db_login['password'], database=self._db_login['database'])
        self._mysql_engine = sa.create_engine(db_conn, echo=False)
        self._mysql_metadata = sa.MetaData()

    def query_table_column_names(self, conn, table: str) -> list:
        cols_query = text(
            """SELECT `COLUMN_NAME` FROM `INFORMATION_SCHEMA`.`COLUMNS`
               WHERE `TABLE_SCHEMA`='{}' AND `TABLE_NAME`=:table """.format(self._db_login['database']))
        query_result = conn.execute(cols_query, table=table)
        db_cols = [x[0] for x in query_result]
        cols = [self._db_dict[c] if c in self._db_dict else c for c in db_cols]
        return cols

    def query_whole_table(self, conn, table: str) -> pd.DataFrame:
        cols = self.query_table_column_names(conn, table)
        query_result = conn.execute(text("""select * from {}""".format(table)))
        df = pd.DataFrame(query_result, columns=cols)
        return df

    def query_last_trading_day(self, conn):
        result = conn.execute(text(
            """SELECT {t} FROM {tbl} ORDER BY {t} DESC  LIMIT 1""".format(
                t=self.TICK_TIME, tbl=self.DB_HIST_1D)))
        last_trading_day = list(result)[0][0]
        return last_trading_day

    def query_trading_dates(self, conn, calendar_days):
        """
        Read trading dates from historical 1-day table within the given calendar days.
        """
        last_trading_day = self.query_last_trading_day(conn)
        earliest = last_trading_day - timedelta(days=calendar_days)
        query = text(
            """SELECT {t} FROM {table} WHERE {t} BETWEEN :earliest AND :last_trading_day""".format(
                t=self.TICK_TIME, table=self.DB_HIST_1D
            ))
        query_result = conn.execute(
            query, earliest=earliest.astimezone(UTC), last_trading_day=last_trading_day.astimezone(UTC))
        trading_dates = sorted(list(set(d[0] for d in query_result)))
        for d in trading_dates:
            if d.strftime('%Y-%m-%d') in INVALID_DATES:
                trading_dates.remove(d)
        return trading_dates

    @staticmethod
    def _map_symbol_ids(specs_df):
        symbol_to_id = {}
        ticker_to_symbol = collections.defaultdict(set)
        ticker_to_id = collections.defaultdict(set)
        id_to_ticker = {}
        id_to_symbol = {}
        for symbol_id, ins in specs_df.iterrows():
            symbol = ins[SYMBOL].upper()
            ticker = re.sub("[0-9]", "", symbol).upper()
            if ticker != '':
                symbol_to_id[symbol] = symbol_id
                ticker_to_id[ticker].add(symbol_id)
                ticker_to_symbol[ticker].add(symbol)
                id_to_ticker[symbol_id] = ticker
                id_to_symbol[symbol_id] = symbol
        return symbol_to_id, ticker_to_symbol, ticker_to_id, id_to_ticker, id_to_symbol

    def _create_symbol_filter(self, symbols):
        if not symbols:
            symbol_filter = ""
        else:
            if isinstance(symbols, str):
                symbols = [symbols,]
            symbols_list = sum(([x.upper(), x.lower()] for x in symbols), [])
            symbol_filter = "{} IN {}".format(self.SYMBOL, tuple(symbols_list))
        return symbol_filter

    def query_specs(self, symbols=None):
        with self._mysql_engine.connect() as conn:
            cols = self.query_table_column_names(conn, self.DB_SPECS)
            symbol_filter = self._create_symbol_filter(symbols)
            if not symbol_filter:
                specs_df = self.query_whole_table(conn, self.DB_SPECS)
            else:
                query = text("""SELECT * FROM {tbl} WHERE {filter}""".format(
                    tbl=self.DB_SPECS, filter=symbol_filter))
                query_result = list(conn.execute(query))
                specs_df = pd.DataFrame(query_result, columns=cols)
        specs_df.set_index(SYMBOL_ID, drop=True, inplace=True)
        specs_df[SYMBOL] = specs_df[SYMBOL].str.upper()
        symbol_to_id, ticker_to_symbol, ticker_to_id, id_to_ticker, id_to_symbol = DataPort._map_symbol_ids(specs_df)
        return specs_df, symbol_to_id, ticker_to_symbol, ticker_to_id, id_to_ticker, id_to_symbol

    def _create_symbol_id_filter(self, symbol_ids):
        if not symbol_ids:
            id_filter = ""
        elif isinstance(symbol_ids, int):
            id_filter = "{}={}".format(self.SYMBOL_ID, symbol_ids)
        elif len(symbol_ids) == 1:
            id_filter = "{}={}".format(self.SYMBOL_ID, symbol_ids[0])
        else:
            id_filter = "{} IN {}".format(self.SYMBOL_ID, tuple(symbol_ids))
        return id_filter

    def query_margin_comm_ratio(self, account_id, symbol_ids=None):
        with self._mysql_engine.connect() as conn:
            cols = self.query_table_column_names(conn, self.DB_MARGIN_COMM)
            symbol_id_filter = self._create_symbol_id_filter(symbol_ids)
            if symbol_id_filter:
                symbol_id_filter += ' AND'
            query = text(
                """SELECT * FROM {tbl} WHERE {filter} account_id={account_id}""".format(
                    tbl=self.DB_MARGIN_COMM, filter=symbol_id_filter, account_id=account_id))
            query_result = list(conn.execute(query))
        margin_comm_ratio_df = pd.DataFrame(query_result, columns=cols).set_index(SYMBOL_ID, drop=True)
        return margin_comm_ratio_df

    def get_timestamp_format(self, hist_data_table):
        if hist_data_table in (self.DB_HIST_1D,):  # daily data table
            time_fmt = self._db_ticktime_format[K_DAY_FMT]
        elif hist_data_table in (self.DB_HIST_1H,):  # hourly data table
            time_fmt = self._db_ticktime_format[K_HOUR_FMT]
        elif hist_data_table in (self.DB_HIST_1M, self.DB_HIST_5M, self.DB_HIST_15M, self.DB_HIST_30M,):
            time_fmt = self._db_ticktime_format[K_MIN_FMT]
        elif hist_data_table in (self.DB_HIST_1S,):  # second data table
            time_fmt = self._db_ticktime_format[K_SEC_FMT]
        elif hist_data_table == self.DB_HIST_TICK:  # tick data table
            time_fmt = self._db_ticktime_format[TICK_FMT]
        else:
            time_fmt = self._db_ticktime_format[K_MIN_FMT]  # default to minute format
        return time_fmt

    def query_hist_data(self, hist_data_table: str, symbol_ids=None, start=None, end=None, min_volume=None):
        """
        start, end: datetime.datetime with pytz.timezone
        """
        if hist_data_table not in (
            self.DB_HIST_1D, self.DB_HIST_1H, self.DB_HIST_30M, self.DB_HIST_15M, self.DB_HIST_5M, self.DB_HIST_1M,
            self.DB_HIST_1S, self.DB_HIST_TICK):
            return
        with self._mysql_engine.connect() as conn:
            cols = self.query_table_column_names(conn, hist_data_table)
            symbol_id_filter = self._create_symbol_id_filter(symbol_ids)
            if symbol_id_filter:
                symbol_id_filter += ' AND'
            time_fmt = self.get_timestamp_format(hist_data_table)
            start = datetime(1980, 1, 1) if start is None else start
            end = datetime(2050, 12, 31) if end is None else end
            if hist_data_table != self.DB_HIST_1D:
                start, end = start.astimezone(UTC), end.astimezone(UTC)
            start, end = start.strftime(time_fmt), end.strftime(time_fmt)
            min_volume_switch = "" if min_volume is None else "AND {volume}>={min_volume}".format(
                volume=self.VOLUME, min_volume=min_volume)
            query = text(
                """SELECT * FROM {tbl} WHERE {id_filter} {t} BETWEEN "{start}" AND "{end}" {volume_filter}
                """.format(
                    tbl=hist_data_table, id_filter=symbol_id_filter, t=self.TICK_TIME, start=start, end=end,
                    volume_filter=min_volume_switch))
            query_result = list(conn.execute(query))
        data_df = pd.DataFrame(query_result, columns=cols)
        data_df = data_df.set_index(TICK_TIME, drop=True)   
        for invalid_date in INVALID_DATES:
            try:
                drop_index = data_df.loc[invalid_date].index
            except KeyError:
                continue
            if not drop_index.empty:
                data_df = data_df.drop(drop_index)
        return data_df

    def query_1month_data(self, symbol_ids, hist_data_table=None):
        if hist_data_table is None:
            hist_data_table = self.DB_HIST_1M
        with self._mysql_engine.connect() as conn:            
            trading_dates = self.query_trading_dates(conn, calendar_days=38)
        start_day, end_day = trading_dates[0], trading_dates[-1]
        n1 = start_day - timedelta(days=1) # start day back shift 1 for previous day close price
        start_datetime = SHANGHAI.localize(datetime(n1.year, n1.month, n1.day, 16, 0, 0))
        end_datetime = SHANGHAI.localize(datetime(end_day.year, end_day.month, end_day.day, 16, 0, 0))
        query_start_time = time_mod.time()
        data_m = self.query_hist_data(hist_data_table, symbol_ids, start=start_datetime, end=end_datetime)
        self._logger.debug("DB query time: {} seconds".format(time_mod.time() - query_start_time))
        return data_m, trading_dates

    def split_data_per_symbol(self, data_df: pd.DataFrame, id_to_symbol: dict) -> dict:
        symbol_dfs = {}
        all_symbol_ids = set(data_df[SYMBOL_ID])
        for symbol_id in all_symbol_ids:
            symbol = id_to_symbol[symbol_id] 
            symbol_df = data_df[data_df[SYMBOL_ID]==symbol_id]
            symbol_dfs[symbol] = symbol_df
        return symbol_dfs


if __name__ == '__main__':
    import sys
    from .china_globals import ALPHAMONGO_DB_DICT, ALPHAMONGO_DB_TICKTIME_FORMAT

    db_login = {'host': '', 'port': 3306, 'user': '', 'password': '', 'database': ''}
    dp = DataPort(db_login, ALPHAMONGO_DB_DICT, ALPHAMONGO_DB_TICKTIME_FORMAT)

    specs,symbol2id, _,_,_,_ = dp.query_specs(('rb1910','i1910','m1909','rm909'))

    with dp._mysql_engine.connect() as conn:
        t_dates = dp.query_trading_dates(conn, 50)
  
    df = dp.query_hist_data('instrument_history_d', specs.index.to_list(), t_dates[9], t_dates[15])
