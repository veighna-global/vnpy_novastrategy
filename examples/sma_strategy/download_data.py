from datetime import datetime
from time import sleep

from vnpy.trader.database import get_database
from vnpy.event import EventEngine
from vnpy.trader.object import HistoryRequest
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.event import EVENT_LOG

from vnpy_binance import BinanceLinearGateway


def output_log(event):
    log = event.data
    print(log.time, log.msg)


if __name__ == "__main__":
    ee = EventEngine()
    ee.register(EVENT_LOG, output_log)
    ee.start()

    setting = {
        "API Key": "",
        "API Secret": "",
        "Server": "REAL",
        "Kline Stream": "True",
        "Proxy Host": "localhost",
        "Proxy Port": 1081
    }

    gateway = BinanceLinearGateway(ee, BinanceLinearGateway.default_name)
    gateway.connect(setting)

    sleep(10)

    req = HistoryRequest(
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
        interval=Interval.MINUTE,
        start=datetime(2022, 1, 1),
        end=datetime(2024, 12, 31)
    )
    bars = gateway.query_history(req)
    print(bars[0])
    print(bars[-1])

    db = get_database()
    db.save_bar_data(bars)
    print("Saved", len(bars))

    gateway.close()
    ee.stop()
