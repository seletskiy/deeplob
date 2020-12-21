import os
import sys
import datetime
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig(
    format='%(asctime)-15s (%(module)s) [%(levelname)s] %(message)s'
)

import lib
import bot

from ibapi.order import Order
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

symbol = sys.argv[1]
exchange = sys.argv[2]
model_path = sys.argv[3]

class Disconnected(Exception):
    pass

class App(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.done = False

    def start(self):
        logging.info("START")

        contract = Contract()
        contract.localSymbol = symbol
        contract.secType = 'FUT'
        contract.exchange = exchange
        contract.currency = 'USD'

        self.reqMktDepth(2001, contract, 10, False, None);

    def keyboardInterrupt(self):
        self.done = True
        self.disconnect()

    def updateMktDepth(self, reqId , level, operation, side, price, size):
        order = self.bot.tick(
            datetime.datetime.utcnow(),
            level,
            operation,
            side,
            price,
            size
        )
        if order is None:
            return

        if order.amount == 0:
            # todo cancel
        else:
            self.trade(order.id, order.amount, order.price)

    def trade(self, id, amount, price):
        order = Order()
        order.totalQuantity = abs(amount)
        if price == 0:
            order.orderType = "MKT"
        else:
            # todo

        if amount > 0:
            order.action = "BUY"
            # order.lmtPrice = round(self.bot.ask(), 2)
        else:
            order.action = "SELL"
            # order.lmtPrice = round(self.bot.bid(), 2)

        contract = Contract()
        contract.localSymbol = symbol
        contract.secType = 'FUT'
        contract.exchange = exchange
        contract.currency = 'USD'

        self.placeOrder(id, contract, order)

    def execDetails(self, reqId, contract, execution):
        super().execDetails(reqId, contract, execution)

    def connectionClosed(self):
        raise Disconnected()

    def error(self, reqId, code, error):
        if reqId > 0:
            # Market depth data has been RESET.
            # Please empty deep book contents before applying any new entries.
            if code == 317:
                self.bot.reset()
            else:
                logging.error("[%d] %s", code, error)
                self.disconnect()

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.bot = bot.Bot(depth=10, model_path=model_path, order_id=orderId)
        self.thread = threading.Thread(target=self.start)
        self.thread.start()

# def orderStatus(self, orderId: OrderId, status: str, filled: float,
#                 remaining: float, avgFillPrice: float, permId: int,
#                 parentId: int, lastFillPrice: float, clientId: int,
#                 whyHeld: str, mktCapPrice: float):


app = App()

while not app.done:
    # try:
        app.connect('127.0.0.1', 4003, 1773)
        app.run()
    # except Disconnected:
    #     time.sleep(1)
