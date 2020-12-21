import os
import sys
import datetime
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig(
    format='%(asctime)-15s (%(module)s) [%(levelname)s] %(message)s'
)

class ShutdownHandler(logging.Handler):
    def emit(self, record):
        print(record.msg, file=sys.stderr)
        logging.shutdown()
        sys.exit(1)

logging.getLogger().addHandler(ShutdownHandler(level=50))

import lib
import bot


class Portfolio():
    def __init__(self):
        self.position = 0
        self.price = 0
        self.profit = 0
        self.trades = 0
        self.wins = 0

    def fee(self, fee):
        self.profit -= fee

    def add(self, position, price):
        if self.position == 0:
            self.position = position
            self.price = price
        else:
            profit = (price - self.price) * self.position

            self.profit += profit
            self.position = 0
            self.price = 0
            self.trades += 1
            if profit > 0:
                self.wins += 1


class Market():
    def __init__(self, delay=datetime.timedelta(milliseconds=700)):
        self.orders  = []
        self.cancels = []
        self.delay = delay
        self.portfolio = Portfolio()
        self.order_id = 1
        self.timestamp = None

    def next_order_id(self):
        order_id = self.order_id
        self.order_id += 1
        return order_id

    def place(self, order: bot.Order):
        order.timestamp = self.timestamp

        if isinstance(order, bot.Cancel):
            logging.info("{%s} Cancel: %s", self.timestamp, order)
            self.cancels = [order]
        else:
            logging.info("{%s} Order: %s", self.timestamp, order)
            self.orders += [order]

    def execute(self, timestamp, order: bot.Order, price):
        if order.amount < 0:
            fee = 2.50
        else:
            fee = 1.30

        self.portfolio.add(order.amount, price)
        self.portfolio.fee(fee)

        return bot.Event(
            bot.Event.Status.FILLED,
            id=order.id,
            timestamp=timestamp,
            price=price,
            fee=fee,
            amount=order.amount
        )

    def cancel(self, timestamp, order: bot.Order):
        return bot.Event(
            bot.Event.Status.CANCELED,
            id=order.id,
            timestamp=timestamp
        )

    def tick(self, timestamp, bid, ask):
        def settle(order):
            if order.timestamp + self.delay > timestamp:
                return 0

            if isinstance(order, bot.Limit):
                if order.amount < 0:
                    return bid if order.limit <= bid else 0
                else:
                    return ask if order.limit >= ask else 0
            if isinstance(order, bot.Market):
                if order.amount < 0:
                    return bid
                else:
                    return ask

        self.timestamp = timestamp

        events = []
        orders = []
        cancels = []

        canceled = []

        for cancel in self.cancels:
            if cancel.timestamp + self.delay > timestamp:
                cancels += [cancel]
                continue

            found = False
            for order in self.orders:
                if order.id == cancel.id:
                    canceled += [cancel.id]
                    events += [self.cancel(timestamp, order)]
                    break

            if not found:
                if len(cancels) != 0:
                    logging.fatal("Cancel for non-existing order: %s" % (cancel.id))

        self.cancels = cancels

        for order in self.orders:
            if order.id in canceled:
                continue

            price = settle(order)
            if price == 0:
                orders += [order]
            else:
                events += [self.execute(timestamp, order, price)]
                logging.info(
                    "{%s} Settle: #%d %+d @ %.2f | A %.2f - %.2f B" % (
                        timestamp,
                        order.id,
                        order.amount,
                        price,
                        ask,
                        bid,
                    )
                )

        self.orders = orders

        return events


symbol = sys.argv[1]
market = sys.argv[2]
model_path = sys.argv[3]

market = Market()
strategy = bot.Strategy(market=market)
trader = bot.Bot(depth=10, model_path=model_path, strategy=strategy)

for line in open(sys.argv[4]):
    if line.startswith("E"):
        print(line, file=sys.stderr)
        continue

    if line.startswith("R"):
        book, index = init()
        continue

    try:
        [_, date, time, level, operation, side, price, size] = line.strip().split(" ")
    except:
        print('E malformed line: %s' % line.strip(), file=sys.stderr)
        continue

    if time < '14:35' or time > '20:55':
        continue

    if len(time) == len('00:00:00'):
        time += '.0'

    if operation == 'U':
        operation = 1
    elif operation == 'D':
        operation = 2
    elif operation == 'I':
        operation = 0
    else:
        print('E unknown operation: %s' % operation, file=sys.stderr)
        continue

    if side == 'A':
        side = 0
    elif side == 'B':
        side = 1
    else:
        print('E unknown side: %s' % side, file=sys.stderr)
        continue

    level = int(level)
    price = float(price)
    size = int(size)

    timestamp = datetime.datetime.strptime(
        date + ' ' + time, '%Y-%m-%d %H:%M:%S.%f'
    )

    events = market.tick(timestamp, trader.bid(), trader.ask())

    for event in events:
        logging.info("{%s} Event: %s", timestamp, event)
        trader.event(event)

        if market.portfolio.trades > 0:
            logging.info("PnL: %+.2f %.0f%% %d" % (
                market.portfolio.profit,
                100.0 * market.portfolio.wins / market.portfolio.trades,
                market.portfolio.trades
            ))

    trader.tick(timestamp, level, operation, side, price, size)
    # if order is not None:
    #     if order.amount == 0:
    #         logging.info("{%s} Cancel: %s", timestamp, order)
    #     else:
    #         logging.info("{%s} Order: %s", timestamp, order)

    #     market.place(order, timestamp)
