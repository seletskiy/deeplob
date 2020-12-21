import sys
import time
import jsonpickle
import logging
import threading
import datetime

from enum import Enum

import numpy as np
import pandas as pd
import tensorflow as tf

from timeit import default_timer as timer

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

symbol = sys.argv[1]
exchange = sys.argv[2]
model_path = sys.argv[3]

class BookSide():
    def __init__(self, depth):
        self.levels = [(0, 0) for i in range(depth)]
        self.index = [0 for i in range(depth)]
        self.depth = depth

    def insert(self, level, price, size):
        self.levels[level] = (price, size)
        self.index[level] = 1

    def top(self):
        return self.levels[0][0]

    def remove(self, level):
        self.insert(level, 0, 0)

    def ready(self):
        return sum(self.index) == self.depth

    def __getitem__(self, i):
        return self.levels[i]

class Book():
    def __init__(self, depth):
        self.depth = depth
        self.asks = BookSide(depth)
        self.bids = BookSide(depth)

    def ready(self):
        return self.asks.ready() and self.bids.ready()

    def slice(self):
        slice = [0 for i in range(self.depth*2*2)]

        for i in range(self.depth):
            slice[4*i+0] = self.asks[i][0]
            slice[4*i+1] = self.asks[i][1]
            slice[4*i+2] = self.bids[i][0]
            slice[4*i+3] = self.bids[i][1]

        return slice

class Normalizer():
    def __init__(self):
        self.avg_p = 0
        self.avg_v = 0
        self.std_p = 0
        self.std_v = 0
        pass

    def update(self, sample):
        self.avg_p = sample[:, 0::2].mean()
        self.avg_v = sample[:, 1::2].mean()

        self.std_p = sample[:, 0::2].std(ddof=1)
        self.std_v = sample[:, 1::2].std(ddof=1)

    def normalize(self, row):
        row = row.copy()
        row[0::2] = (row[ 0::2] - self.avg_p) / self.std_p
        row[1::2] = (row[ 1::2] - self.avg_v) / self.std_v
        return row


class Order():
    def __init__(self, id, amount):
        self.id = id
        self.timestamp = None
        self.amount = amount
        # self.price = price

    def __str__(self):
        return "#%d %+d" % (self.id, self.amount)


class Market(Order):
    pass


class Cancel(Order):
    def __init__(self, id):
        super().__init__(id, 0)


class Limit(Order):
    def __init__(self, id, amount, limit):
        super().__init__(id, amount)
        self.limit = limit


    def __str__(self):
        return "%s @ %.2f" % (
            super().__str__(),
            self.limit
        )


class LimitIfTouched(Limit):
    def __init__(self, id, amount, limit, trigger):
        super().__init__(id, amount, limit)

        self.trigger = trigger

    def __str__(self):
        return "%s ^ %.2f [LIT]" % (
            super().__str__(),
            self.trigger,
        )


class Event():
    class Status(Enum):
        FILLED   = 100
        CANCELED = 200

    def __init__(self, status, id, timestamp, price=0, fee=0, amount=0):
        self.status = status
        self.id = id
        self.timestamp = timestamp
        self.price = price
        self.fee = fee
        self.amount = amount

    def __str__(self):
        if self.status == Event.Status.FILLED:
            return "#%d %s @ %.2f (-%.2f)" % (self.id, self.status, self.price, self.fee)

        if self.status == Event.Status.CANCELED:
            return "#%d %s" % (self.id, self.status)

        return "Event: ?"


class Strategy():
    class State(Enum):
        IDLE = 0
        OPENING_SHORT = -1
        OPENING_LONG  = +1
        IDLE_SHORT = -2
        IDLE_LONG  = +2
        CLOSING_SHORT_A = -30
        CLOSING_LONG_A  = +30
        CLOSING_SHORT_B = -31
        CLOSING_LONG_B  = +31
        CANCELING_SHORT = -4
        CANCELING_LONG  = +4

    def __init__(self, market):
        self.bear = 0
        self.bull = 0
        self.neut = 0
        self.state = Strategy.State.IDLE
        self.market = market
        # self.order_id = order_id
        self.order_opening = None
        self.order_take_profit = None
        self.lever = 20

    def event(self, event):
        if event.status == Event.Status.FILLED:
            if self.order_opening is not None and event.id == self.order_opening.id:
                if self.state == Strategy.State.OPENING_LONG:
                    self.switch(event.timestamp, Strategy.State.IDLE_LONG)
                if self.state == Strategy.State.OPENING_SHORT:
                    self.switch(event.timestamp, Strategy.State.IDLE_SHORT)

                self.order_opening = None

            if self.order_take_profit is not None and event.id == self.order_take_profit.id:
                if self.state == Strategy.State.IDLE_LONG:
                    self.switch(event.timestamp, Strategy.State.IDLE)
                if self.state == Strategy.State.IDLE_SHORT:
                    self.switch(event.timestamp, Strategy.State.IDLE)

            if self.state == Strategy.State.CLOSING_LONG_A:
                self.switch(event.timestamp, Strategy.State.IDLE)
            if self.state == Strategy.State.CLOSING_SHORT_A:
                self.switch(event.timestamp, Strategy.State.IDLE)

            # if self.state == Strategy.State.CLOSING_LONG_B:
            #     self.switch(event.timestamp, Strategy.State.IDLE)
            # if self.state == Strategy.State.CLOSING_SHORT_B:
            #     self.switch(event.timestamp, Strategy.State.IDLE)

            # TODO
            if self.state == Strategy.State.CANCELING_LONG:
                self.switch(event.timestamp, Strategy.State.IDLE_LONG)
            if self.state == Strategy.State.CANCELING_SHORT:
                self.switch(event.timestamp, Strategy.State.IDLE_SHORT)

        if event.status == Event.Status.CANCELED:
            if self.state == Strategy.State.CANCELING_LONG:
                self.switch(event.timestamp, Strategy.State.IDLE)
            if self.state == Strategy.State.CANCELING_SHORT:
                self.switch(event.timestamp, Strategy.State.IDLE)

            self.bear = 0
            self.bull = 0
            self.neut = 0

            self.order_opening = None
            self.order_take_profit = None

    def switch(self, timestamp, state):
        logging.info("{%s} Strategy State: %s -> %s" % (timestamp, self.state, state))
        self.state = state

    def act(self, timestamp, bid, ask):
        def order_(amount, price):
            self.order_opening = Market(
                self.market.next_order_id(),
                amount * self.lever,
                # price
            )

            self.market.place(self.order_opening)

            self.order_take_profit = Limit(
                self.market.next_order_id(),
                -amount * self.lever,
                price + amount * 1.5
            )

            self.market.place(self.order_take_profit)

        def cancel_():
            if self.order_opening is not None:
                self.market.place(Cancel(self.order_opening.id))

            self.market.place(Cancel(self.order_take_profit.id))

        def close_(amount):
            order = Market(
                self.market.next_order_id(),
                amount * self.lever,
            )

            self.market.place(order)

        action = 0

        if self.bear >= 20:
            action = -1
        if self.bull >= 20:
            action = +1

        if self.state == Strategy.State.IDLE:
            if action > 0:
                self.switch(timestamp, Strategy.State.OPENING_LONG)
                return order_(+1, ask)
            if action < 0:
                self.switch(timestamp, Strategy.State.OPENING_SHORT)
                return order_(-1, bid)

        if self.state == Strategy.State.IDLE_LONG:
            if self.bear >= 20:
                self.switch(timestamp, Strategy.State.CLOSING_LONG_A)
                return [cancel_(), close_(-1)]
                # return order_(-1, bid)

        if self.state == Strategy.State.IDLE_SHORT:
            if self.bull >= 20:
                self.switch(timestamp, Strategy.State.CLOSING_SHORT_A)
                # return order_(+1, ask)
                return [cancel_(), close_(+1)]

        if self.state == Strategy.State.OPENING_LONG:
            if self.bear >= 20:
                self.switch(timestamp, Strategy.State.CANCELING_LONG)
                return cancel_()

        if self.state == Strategy.State.OPENING_SHORT:
            if self.bull >= 20:
                self.switch(timestamp, Strategy.State.CANCELING_SHORT)
                return cancel_()

    def tick(self, timestamp, bid, ask, prediction):
        dir = np.argmax(prediction)

        if dir == 0:
            self.bear += 1
            self.bull  = 0
            self.neut  = 0
        if dir == 1:
            self.bear  = 0
            self.bull  = 0
            self.neut += 1
        if dir == 2:
            self.bear  = 0
            self.bull += 1
            self.neut  = 0

        # logging.debug(
        #     "{%s} Strategy Tick: %s | %4d %4d %4d | A %.2f - %.2f B = %.2f M" % (
        #         timestamp,
        #         "D-U"[dir],
        #         self.bull,
        #         self.neut,
        #         self.bear,
        #         ask,
        #         bid,
        #         (bid + ask) / 2.0
        #     )
        # )

        return self.act(timestamp, bid, ask)


class Bot():
    class State(Enum):
        INIT_BOOK = 0
        INIT_SAMPLE = 1
        INIT_SEQUENCE = 2
        READY = 4

    def __init__(
        self,
        depth,
        model_path,
        strategy,
        sample_rate=10,
        sequence_window=100,
        normalize_window=1000,
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.depth = depth
        self.sample_nr = 0
        self.sample_rate = sample_rate
        self.sample_size = 0
        self.sequence_size = 0
        self.sequence_window = sequence_window
        self.normalize_window = normalize_window
        self.strategy = strategy
        self.reset()

    def reset(self):
        self.state = Bot.State.INIT_BOOK
        self.book = Book(self.depth)
        self.sample = np.zeros(shape=(self.normalize_window, self.depth * 4))
        self.sequence = np.zeros(shape=(self.sequence_window, self.depth * 4))
        self.normalizer = Normalizer()

    def mid(self):
        return (self.book.asks.top() + self.book.bids.top())/2.0

    def ask(self):
        return self.book.asks.top()

    def bid(self):
        return self.book.bids.top()

    def update_book(self, level, operation, side, price, size):
        side = [self.book.asks, self.book.bids][side]

        if operation == 0 or operation == 1:
            side.insert(level, price, size)

        if operation == 2:
            side.remove(level)

        return self.book.ready()

    def update_sample(self):
        self.sample_nr += 1

        if self.sample_nr % self.sample_rate == 0:
            self.sample_nr = 0

            self.sample = np.roll(self.sample, -1, axis=0)
            self.sample[-1] = self.book.slice()
            self.sample_size += 1

            if self.sample_size >= self.normalize_window:
                self.sample_size = self.normalize_window
                return True
            else:
                return False

        return False

    def update_sequence(self):
        self.sequence = np.roll(self.sequence, -1, axis=0)

        self.normalizer.update(self.sample)
        self.sequence[-1] = self.normalizer.normalize(self.sample[-1])
        self.sequence_size += 1

        if self.sequence_size >= self.sequence_window:
            return True
        else:
            return False

    def tick(self, timestamp, level, operation, side, price, size):
        if not self.update(timestamp, level, operation, side, price, size):
            return

        return self.strategy.tick(
            timestamp,
            self.bid(),
            self.ask(),
            self.predict()
        )

    def event(self, event):
        self.strategy.event(event)

    def predict(self):
        def predict_():
            input = self.sequence.copy().reshape(1, self.sequence_window, self.depth * 4, 1)
            return self.model.predict_on_batch(input)

        # start = timer()
        prediction = predict_()
        # end = timer()

        # logging.info("Prediction Took: %s" % (end - start))

        return prediction

    def update(self, timestamp, level, operation, side, price, size):
        if not self.update_book(level, operation, side, price, size):
            return False

        if self.state == Bot.State.INIT_BOOK:
            self.state = Bot.State.INIT_SAMPLE
            logging.info("{%s} L2 Book Ready" % timestamp)

        if not self.update_sample():
            if self.sample_nr == 0 and self.sample_size % 10 == 0:
                logging.info(
                    "{%s} Sample Status: %d of %d" % (
                        timestamp,
                        self.sample_size,
                        self.normalize_window))
            return False

        if self.state == Bot.State.INIT_SAMPLE:
            self.state = Bot.State.INIT_SEQUENCE
            logging.info("{%s} Sample Ready: %d" % (
                timestamp,
                self.normalize_window
            ))

        if not self.update_sequence():
            if self.sample_nr == 0 and self.sequence_size % 10 == 0:
                logging.info(
                    "{%s} Main Sequence Status: %d of %d" % (
                        timestamp,
                        self.sequence_size,
                        self.sequence_window))
            return False

        if self.state == Bot.State.INIT_SEQUENCE:
            self.state = Bot.State.READY
            logging.info("{%s} Main Sequence Ready: %d" % (
                timestamp,
                self.sequence_window
            ))

        return True
