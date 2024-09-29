import pandas as pd
import talib as ta

from vnpy_novastrategy import (
    StrategyTemplate,
    BarData, TickData,
    TradeData, OrderData,
    Interval, datetime,
    DataTable, round_to
)


class TrendStrategy(StrategyTemplate):
    """Classic turtle-trading strategy"""

    author: str = "VeighNa Global"

    boll_window: int = 20
    boll_dev: int = 2
    atr_window: int = 14
    trailing_multiplier: float = 2
    risk_level: float = 5000
    percent_add: float = 0.01

    trading_size: float = 0.0
    trading_target: float = 0.0
    trading_pos: float = 0.0
    boll_up: float = 0.0
    boll_down: float = 0.0
    intra_trade_high: float = 0.0
    intra_trade_low: float = 0.0
    long_stop: float = 0.0
    short_stop: float = 0.0

    parameters = [
        "boll_window",
        "boll_dev",
        "atr_window",
        "risk_level"
    ]
    variables = [
        "trading_size",
        "trading_target",
        "trading_pos",
        "boll_up",
        "boll_down",
        "intra_trade_high",
        "intra_trade_low",
        "long_stop",
        "short_stop",
    ]

    def on_init(self) -> None:
        """Callback when strategy is inited"""
        self.agg_setting: dict = {
            "open_price": "first",
            "high_price": "max",
            "low_price": "min",
            "close_price": "last"
        }
        self.trading_symbol: str = self.vt_symbols[0]

        self.table: DataTable = DataTable(
            vt_symbols=[self.trading_symbol],
            size=60 * 100,
            interval=Interval.MINUTE
        )

        self.load_bars(10, Interval.MINUTE)

        self.write_log("Strategy is inited.")

    def on_start(self) -> None:
        """Callback when strategy is started"""
        self.write_log("Strategy is started.")

    def on_stop(self) -> None:
        """Callback when strategy is stoped"""
        self.write_log("Strategy is stopped.")

    def on_tick(self, tick: TickData) -> None:
        """Callback of tick data update"""
        bar: BarData = tick.extra.get("bar", None)
        if not bar:
            return

        bars: dict = {bar.vt_symbol: bar}
        self.on_bars(bars)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """Callback of 1-minute candle bars update"""
        # Update bars into data table
        self.table.update_bars(bars)
        if not self.table.inited:
            return

        # Check if hour finished
        last_dt: datetime = self.table.get_dt()
        if last_dt.minute != 59:
            return

        # Aggregate hour OHLC data
        minute_df: pd.DataFrame = self.table.get_df().xs(self.trading_symbol, level=1)
        hour_df: pd.DataFrame = minute_df.resample("h").agg(self.agg_setting)

        # Calculate technical indicator
        sma_s: pd.Series = ta.SMA(hour_df["close_price"], self.boll_window)
        std_s: pd.Series = ta.STDDEV(hour_df["close_price"], self.boll_window)
        boll_up_s: pd.Series = sma_s + std_s * self.boll_dev
        boll_down_s: pd.Series = sma_s - std_s * self.boll_dev
        atr_s: pd.Series = ta.ATR(hour_df["high_price"], hour_df["low_price"], hour_df["close_price"])

        self.boll_up = boll_up_s.iloc[-1]
        self.boll_down = boll_down_s.iloc[-1]
        self.atr_value = atr_s.iloc[-1]

        # Generate trading target position
        bar: BarData = bars[self.trading_symbol]

        if not self.trading_target:
            self.trading_size = round(self.risk_level / self.atr_value, 2)

            if bar.high_price >= self.boll_up:
                self.trading_target = self.trading_size
                self.intra_trade_high = bar.close_price
            elif bar.low_price <= self.boll_down:
                self.trading_target = -self.trading_size
                self.intra_trade_low = bar.close_price

        elif self.trading_target > 0:
            if bar.low_price <= self.long_stop:
                self.trading_target = 0
            else:
                self.intra_trade_high = max(self.intra_trade_high, bar.high_price)
                self.long_stop = self.intra_trade_high - self.atr_value * self.trailing_multiplier

        elif self.trading_target < 0:
            if bar.high_price >= self.short_stop:
                self.trading_target = 0
            else:
                self.intra_trade_low = min(self.intra_trade_low, bar.low_price)
                self.short_stop = self.intra_trade_low + self.atr_value * self.trailing_multiplier

        # Cancel all existing orders
        self.cancel_all()

        # Send new order according to the difference between target and pos
        trading_volume: int = self.trading_target - self.trading_pos
        pricetick: float = self.get_pricetick(self.trading_symbol)

        if trading_volume > 0:
            buy_price: float = round_to(bar.close_price * (1 + self.percent_add), pricetick)
            self.buy(self.trading_symbol, buy_price, abs(trading_volume))
        elif trading_volume < 0:
            short_price: float = round_to(bar.close_price * (1 - self.percent_add), pricetick)
            self.short(self.trading_symbol, short_price, abs(trading_volume))

        # Put event to upgrade GUI
        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        """Callback of trade update"""
        # Get latest pos
        self.trading_pos = self.get_pos(self.trading_symbol)

        # Put event to upgrade GUI
        self.put_event()

    def on_order(self, order: OrderData) -> None:
        """Callback of order update"""
        pass
