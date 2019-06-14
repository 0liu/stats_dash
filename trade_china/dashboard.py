# -*- coding: utf-8 -*-

"""
Trading dashboard.
"""


import sys, os
import time
from datetime import datetime
from .utils import get_logger, get_config
from .constants import SYMBOL, LEG_i_SYMBOL
from .excel_manager import ExcelManager, SpreadsheetNotFoundError
from .rmq_listener import RabbitMQListener
from .dataport import DataPort
from .market_analyzer import MarketAnalyzer
from .china_globals import ALPHAMONGO_DB_DICT, ALPHAMONGO_DB_TICKTIME_FORMAT


class Dashboard:
    PAIR_SINGLE_PARAMS = {
        True: {  # Pairs parameters
            'max_leg': 2,
            'start_row': 1,
            'ma_days': (1, 2, 5, 10, 20),
            'ma_day_level': 2,
            'max_level': 4,
        },
        False: {  # Single contract parameters
            'max_leg': 1,
            'start_row': 1,
            'ma_days': (1, 2, 5, 10, 20),
            'ma_day_level': 2,
            'max_level': 4,
        }
    }

    def __init__(self):
        self.logger = get_logger(datetime.now().strftime('trade_china_%Y-%m-%d_%H-%M-%S'))
        self.logger_instance = None
        self.config = None
        self.pair_or_single = None
        self.pair_single_params = None
        self.excel_file = None
        self.sheet_name = None
        self.account_id = None

        self.ctrl_df = None
        self.study_df = None
        self.data_port = None
        self.market_listener = None
        self.market_analyzer = None

    def reset(self):
        """
        Reload Excel file and re-init objects.
        """
        self.pair_single_params = Dashboard.PAIR_SINGLE_PARAMS[self.pair_or_single]
        self.account_id = self.config['Trading']['account_id']

        self.logger.info("Account ID: {}".format(self.account_id))
        self.logger.info("Single or Pair: {}".format(('Single', 'Pair')[self.pair_or_single]))
        self.logger.info("Excel file: {}  Spreadsheet: {}".format(self.excel_file, self.sheet_name))
        self.logger.info("Parameters: {}".format(self.pair_single_params))

        self.excel_manager = ExcelManager(
            max_leg=self.pair_single_params['max_leg'], excel_file=self.excel_file, sheet_name=self.sheet_name,
            start_row=self.pair_single_params['start_row'], ma_days=self.pair_single_params['ma_days'],
            max_level=self.pair_single_params['max_level'], logger=self.logger)
        self.ctrl_df, self.study_df = self.excel_manager.load_spreadsheet()
        if self.pair_or_single:  # pairs
            self.watch_symbols = {
                s for leg in range(1, self.pair_single_params['max_leg']+1)
                for s in self.ctrl_df[LEG_i_SYMBOL.format(i=leg)] if s is not None}
        else:  # single contracts
            self.watch_symbols = {s for s in self.ctrl_df[SYMBOL] if s is not None}

        self.market_listener = RabbitMQListener(
            rmq_config=self.config['TickData'], binding_keys=("com.#.InstrumentMessage",),
            watch_symbols=self.watch_symbols, logger=self.logger)
        self.data_port = DataPort(
            db_login=self.config['HistData'], db_dict=ALPHAMONGO_DB_DICT,
            db_ticktime_format=ALPHAMONGO_DB_TICKTIME_FORMAT, logger=self.logger)
        self.market_analyzer = MarketAnalyzer(
            max_leg=self.pair_single_params['max_leg'], control_df=self.ctrl_df, study_df=self.study_df,
            msg_queue=self.market_listener.msg_queue, dataport=self.data_port,
            ma_days=self.pair_single_params['ma_days'], max_scalping_level=self.pair_single_params['max_level'],
            ma_day_level=self.pair_single_params['ma_day_level'], logger=self.logger)

        # Reload pair/symbol definitions, control knobs and query symbol specs.
        self.market_analyzer.update_specs()
        self.excel_manager.write_control()

    def reset_config(self):
        self.config = get_config(self.logger, reset=True)
        self.logger.info({s: dict(self.config[s]) for s in self.config.sections()})

    def set_excel_spreadsheet(self):
        """
        User sets Excel file and spreadsheet name, and reload everything.
        """
        if self.market_analyzer and self.market_analyzer._update_study_on_tick_thread_running:
            print("\nYou need to stop streaming market before set and reload spreadsheet.\n")
            return
        while True:
            print("\nPlease select Excel format: 0 - Single contract  1 - Pairs (Default: 1 - Pairs):")
            format_input = input().strip()
            if format_input in ('', '0', '1'):
                self.pair_or_single = {'': True, '0': False, '1': True}[format_input]            
                break
            else:
                print("Wrong input. Please type number 0 for single contract, Return or 1 for pairs.")
                continue
        while True:
            try:
                print("\nPlease input Excel file and spreadsheet names (Return='DailyUpdate.xlsx  Pair', q=Quit):")
                excel_input = input().strip()
                if excel_input.lower() == 'q':
                    self.shutdown()
                elif excel_input == '':
                    self.excel_file, self.sheet_name = 'DailyUpdate.xlsx', 'Pair'
                else:
                    self.excel_file, self.sheet_name = excel_input.split()
                if self.excel_file.split('.')[-1] != 'xlsx':
                    self.excel_file += '.xlsx'
                self.reset()
                print("\n")
                break
            except (FileNotFoundError, SpreadsheetNotFoundError) as e:
                print(e)

    def update_daily_studies(self):
        self.market_analyzer.stop_update_study_on_knobs()
        self.market_analyzer.update_daily_studies(account_id=self.account_id)
        self.excel_manager.write_study()
        self.market_analyzer.start_update_study_on_knobs()

    def start_stop_streaming(self, restart: bool=None):
        if restart is None:
            restart = not self.market_analyzer._update_study_on_tick_thread_running
        self.market_analyzer.stop_update_study_on_knobs()
        self.market_listener.stop_listener()
        self.market_analyzer.stop_update_study_on_tick()
        self.excel_manager.stop_read_control()
        self.excel_manager.stop_update_study()
        if restart:
            self.excel_manager.start_update_study()
            self.excel_manager.start_read_control()
            self.market_analyzer.start_update_study_on_tick()
            self.market_listener.start_listener()
            self.market_analyzer.start_update_study_on_knobs()

    def shutdown(self):
        print("Shutting down...")
        try:
            self.start_stop_streaming(restart=False)
        except AttributeError:
            pass
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    def run(self):
        self.config = get_config(self.logger)
        self.logger.info({s: dict(self.config[s]) for s in self.config.sections()})
        self.set_excel_spreadsheet()
        
        prompt = ("\n"
            "---------------------------------------------\n"
            "Press a number to select a task:\n"
            "  1 - Reset data connection settings.\n"
            "  2 - Set Excel format, file name, spreadsheet names. Load definitions.\n"
            "  3 - Update statistics at the last day close.\n"
            "  4 - Start/Stop streaming market.\n"
            "  0 - Quit.\n"
            "\n")

        while True:
            print(prompt)
            user_select = input().strip()
            if user_select == '':
                continue
            if user_select.split()[0] not in [str(c) for c in range(9)]:
                print("Invalid input. Please select again.")
                continue
            operation_func = {
                '1': self.reset_config,
                '2': self.set_excel_spreadsheet,
                '3': self.update_daily_studies,
                '4': self.start_stop_streaming,
                '0': self.shutdown,
            }[user_select]
            operation_func()


if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard.run()