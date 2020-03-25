# -*- coding: utf-8 -*-

"""
Receive market data and/or trade/order messsages from RabbitMQ publisher.
"""


import time
from io import BytesIO
from threading import Thread, Lock
from queue import Queue
import pika
import avro.schema
import avro.io
from .utils import get_logger
from .china_globals import EXCHANGES_DICT, AVRO_SCHEMAS_PATH, ALPHAMONGO_TICK_DICT


class RabbitMQListener:

    def __init__(
        self, rmq_config, binding_keys=("com.#.InstrumentMessage",),
        tick_dict=ALPHAMONGO_TICK_DICT, message_queue=None, watch_symbols=None, logger=None, test_mode=False
        ):

        self._host = rmq_config['host']
        self._port = int(rmq_config['port'])
        self._user = rmq_config['user']
        self._password = rmq_config['password']
        self._exchange = EXCHANGES_DICT[rmq_config['exchange']]
        self._binding_keys = binding_keys
        self._tick_dict = tick_dict
        self._test_mode = test_mode

        self.msg_queue = Queue() if message_queue is None else message_queue

        self._connection = None
        self._channel = None
        self._queue_name = None
        self._customer_tag = None
        self._schema_dict = {}

        self._message_listener_thread = None
        self._thread_lock = Lock()
        self._is_listening = False  # thread locked

        self._watch_symbols = watch_symbols

        self._logger = logger if logger is not None else get_logger('rmq_listener')

    def _connect_2_rabbit(self):
        parameters = None
        try:
            parameters = pika.ConnectionParameters(
                host=self._host, port=self._port, virtual_host='/', connection_attempts=3,
                retry_delay=30, credentials=pika.PlainCredentials(self._user, self._password))
            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()
            self._channel.exchange_declare(exchange=self._exchange, exchange_type='topic')
            result = self._channel.queue_declare(exclusive=True)
            self._queue_name = result.method.queue
            for binding_key in self._binding_keys:
                self._channel.queue_bind(exchange=self._exchange, queue=self._queue_name, routing_key=binding_key)
            self._logger.info("RabbitMQ connected successfully! Parameters: {}".format(parameters))
        except BaseException as e:
            self._logger.error('RabbitMQ _connect_2_rabbit Error: Parameters: {}'.format(parameters))
            self._logger.error('RabbitMQ _connect_2_rabbit Error: {}'.format(e.__repr__()), exc_info=True)

    def _callback(self, ch, method, properties, body):
        schema_name = method.routing_key.rsplit('.', 1)[-1]
        if schema_name in self._schema_dict:
            schema = self._schema_dict[schema_name]
        else:
            schema = avro.schema.Parse(open('{}/{}.avro'.format(AVRO_SCHEMAS_PATH, schema_name)).read())
            self._schema_dict[schema_name] = schema
        buffer_reader = BytesIO(body)
        buffer_decoder = avro.io.BinaryDecoder(buffer_reader)
        datum_reader = avro.io.DatumReader(schema)
        msg = datum_reader.read(buffer_decoder)
        self._distribute_message(schema_name, msg)

    def _distribute_message(self, schema_name, msg):
        if schema_name == "InstrumentMessage":
            self._subscribe_instrument(msg)
        elif schema_name == "TradeMessage":
            self._subscribe_trade_message(msg)
        elif schema_name == "OrderStatusMessage":
            self._subscribe_order_status_message(msg)
        else:
            msg.clear()

    def _subscribe_instrument(self, instrument_msg):
        """
        Put to message Queue to be read by other threads.
        """
        if self._watch_symbols is not None and instrument_msg['symbol'].upper() not in self._watch_symbols:
            instrument_msg.clear()
            return
        translated_msg = {self._tick_dict[k] if k in self._tick_dict else k: v for k, v in instrument_msg.items()}
        self.msg_queue.put(translated_msg)
        if self._test_mode:
            #if instrument_msg['symbol'].upper() == 'RB1910':
            self._logger.info("\n{}".format(instrument_msg))
            self._logger.info("\n{}".format(self.msg_queue.get()))

    def _subscribe_trade_message(self, trade_msg):
        trade_msg.clear()

    def _subscribe_order_status_message(self, order_status_msg):
        order_status_msg.clear()

    def _start_consuming(self):
        while True:
            self._thread_lock.acquire()
            if not self._is_listening:
                self._thread_lock.release()
                break
            else:
                self._thread_lock.release()

            self._connect_2_rabbit()

            try:
                self._customer_tag = self._channel.basic_consume(self._callback, queue=self._queue_name, no_ack=True)
                self._channel.start_consuming()
            except BaseException as e:
                self._logger.error('RabbitMQ _start_consuming Error: {}'.format(e.__repr__()), exc_info=True)

    def _stop_consuming(self):
        try:
            self._thread_lock.acquire()
            self._is_listening = False
            self._thread_lock.release()

            # Sleep 2 second in case the output Queue interfacing other threads is deleted immediately after it is
            # just created. Some RabbitMQ object may still being constructed but not done yet, and some RabbitMQ
            # call will fail.
            time.sleep(2)

            try:
                self._channel.cancel()
                for binding_key in self._binding_keys:
                    self._channel.queue_unbind(exchange=self._exchange, queue=self._queue_name, routing_key=binding_key)
                self._channel.queue_delete(self._queue_name)
                self._channel.close()
            except AttributeError as e:
                self._logger.error('RabbitMQ _stop_consuming error: {}'.format(e.__repr__()), exc_info=True)
            try:
                self._connection.close()
            except AttributeError as e:
                self._logger.error('RabbitMQ _stop_consuming error: {}'.format(e.__repr__()), exc_info=True)
        except BaseException as e:
            self._logger.error('RabbitMQ _stop_consuming error: {}'.format(e.__repr__()), exc_info=True)

    def start_listener(self):
        self._message_listener_thread = Thread(target=self._start_consuming)
        self._thread_lock.acquire()
        self._is_listening = True
        self._thread_lock.release()
        self._message_listener_thread.start()
        self._logger.info("RabbitMQ listener thread started. Host={}:{}@{}:{}. Exchange={}. BindingKeys={}".format(
            self._user, self._password, self._host, self._port, self._exchange, self._binding_keys
        ))

    def stop_listener(self):
        if self._message_listener_thread is None or not self._message_listener_thread.isAlive():
            return
        self._stop_consuming()
        if self._message_listener_thread.isAlive():
            self._message_listener_thread.join()
        self._logger.info("RabbitMQ listener thread stopped.")


# Unit test
def test():
    import os, sys, importlib
    from .utils import get_logger, get_config
    logger = get_logger('rmq_listener')
    rmq_config = get_config(logger)['TickData']
    listener = RabbitMQListener(rmq_config=rmq_config, test_mode=True, logger=logger)
    listener.start_listener()
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt as e:
            listener._logger.info('\nRabbitMQ listener received keyboard interrupt.\n')
            break

    listener.stop_listener()
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

if __name__ == '__main__':
    test()
