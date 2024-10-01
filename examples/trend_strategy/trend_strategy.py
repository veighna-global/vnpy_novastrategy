import pandas as pd
import talib as ta

from vnpy_novastrategy import (
    StrategyTemplate,
    Parameter, Variable,
    BarData, TickData,
    TradeData, OrderData,
    Interval, DataTable, round_to,
)


class TrendStrategy(StrategyTemplate):
    """Classic turtle-trading strategy"""

    author: str = "VeighNa Global"

    boll_window: int = Parameter(75)
    boll_dev: int = Parameter(5)
    atr_window: int = Parameter(20)
    trailing_multiplier: float = Parameter(6.5)
    risk_level: float = Parameter(5000)

    trading_size: float = Variable(0.0)
    trading_target: float = Variable(0.0)
    trading_pos: float = Variable(0.0)
    trading_allowed: bool = Variable(False)
    boll_up: float = Variable(0.0)
    boll_down: float = Variable(0.0)
    intra_trade_high: float = Variable(0.0)
    intra_trade_low: float = Variable(0.0)
    long_stop: float = Variable(0.0)
    short_stop: float = Variable(0.0)

    def on_init(self) -> None:
        """Callback when strategy is inited"""
        self.trading_symbol: str = self.vt_symbols[0]

        self.table: DataTable = DataTable(
            vt_symbols=[self.trading_symbol],
            size=100,
            window=1,
            interval=Interval.HOUR
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
        # Check breakout and execute trading every minute
        bar: BarData = bars[self.trading_symbol]
        self.check_breakout(bar)
        self.execute_trading(bar)

        # Update bars into data table
        finished: bool = self.table.update_bars(bars)

        # Calculate indicators every hour
        if finished:
            df: pd.DataFrame = self.table.get_df().xs(self.trading_symbol, level=1)

            sma_s: pd.Series = ta.SMA(df["close_price"], self.boll_window)
            std_s: pd.Series = ta.STDDEV(df["close_price"], self.boll_window)
            boll_up_s: pd.Series = sma_s + std_s * self.boll_dev
            boll_down_s: pd.Series = sma_s - std_s * self.boll_dev
            self.boll_up = boll_up_s.iloc[-1]
            self.boll_down = boll_down_s.iloc[-1]

            atr_s: pd.Series = ta.ATR(df["high_price"], df["low_price"], df["close_price"], self.atr_window)
            self.atr_value = atr_s.iloc[-1]
            self.trading_size = round(self.risk_level / self.atr_value, 2)

            self.trading_allowed = True

        # Put event to upgrade GUI
        self.put_event()

    def check_breakout(self, bar: BarData) -> None:
        """Check critical level breakout"""
        if not self.trading_allowed:
            return

        # Holding no position
        if not self.trading_target:
            if bar.high_price >= self.boll_up:
                self.trading_target = self.trading_size
                self.intra_trade_high = bar.close_price

                self.trading_allowed = False
            elif bar.low_price <= self.boll_down:
                self.trading_target = -self.trading_size
                self.intra_trade_low = bar.close_price

                self.trading_allowed = False
        # Holding long position
        elif self.trading_target > 0:
            if bar.low_price <= self.long_stop:
                self.trading_target = 0

                self.trading_allowed = False
            else:
                self.intra_trade_high = max(self.intra_trade_high, bar.high_price)
                self.long_stop = self.intra_trade_high - self.atr_value * self.trailing_multiplier
        # Holding short position
        elif self.trading_target < 0:
            if bar.high_price >= self.short_stop:
                self.trading_target = 0

                self.trading_allowed = False
            else:
                self.intra_trade_low = min(self.intra_trade_low, bar.low_price)
                self.short_stop = self.intra_trade_low + self.atr_value * self.trailing_multiplier

    def execute_trading(self, bar: BarData) -> None:
        """Execute trading according to the difference between target and pos"""
        # Cancel all existing orders
        self.cancel_all()

        # Send new order according to the difference between target and pos
        trading_volume: int = self.trading_target - self.trading_pos
        if not trading_volume:
            return

        pricetick: float = self.get_pricetick(self.trading_symbol)

        if trading_volume > 0:
            buy_price: float = round_to(bar.close_price - pricetick, pricetick)
            self.buy(self.trading_symbol, buy_price, abs(trading_volume))
        elif trading_volume < 0:
            short_price: float = round_to(bar.close_price + pricetick, pricetick)
            self.short(self.trading_symbol, short_price, abs(trading_volume))

    def on_trade(self, trade: TradeData) -> None:
        """Callback of trade update"""
        # Get latest pos
        self.trading_pos = self.get_pos(self.trading_symbol)

        # Put event to upgrade GUI
        self.put_event()

    def on_order(self, order: OrderData) -> None:
        """Callback of order update"""
        pass
