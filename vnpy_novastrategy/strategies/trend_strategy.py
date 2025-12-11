from typing import cast

import numpy as np
import talib as ta
import pandas as pd

from vnpy_novastrategy import (
    StrategyTemplate,
    Parameter, Variable,
    BarData, TickData,
    TradeData, OrderData,
    Interval,
    DataTable, TickHandler,
)


class TrendStrategy(StrategyTemplate):
    """Classic turtle-trading strategy"""

    author: str = "VeighNa Global"

    boll_window: Parameter[int] = Parameter(75)
    boll_dev: Parameter[int] = Parameter(5)
    atr_window: Parameter[int] = Parameter(20)
    trailing_multiplier: Parameter[float] = Parameter(6.5)
    risk_level: Parameter[float] = Parameter(5000.0)

    trading_size: Variable[float] = Variable(0.0)
    trading_target: Variable[float] = Variable(0.0)
    trading_pos: Variable[float] = Variable(0.0)
    trading_allowed: Variable[bool] = Variable(False)
    boll_up: Variable[float] = Variable(0.0)
    boll_down: Variable[float] = Variable(0.0)
    intra_trade_high: Variable[float] = Variable(0.0)
    intra_trade_low: Variable[float] = Variable(0.0)
    long_stop: Variable[float] = Variable(0.0)
    short_stop: Variable[float] = Variable(0.0)

    def on_init(self) -> None:
        """Callback when strategy is inited"""
        self.trading_symbol: str = self.vt_symbols[0]

        self.handler: TickHandler = TickHandler(
            vt_symbols=self.vt_symbols,
            on_bars=self.on_bars
        )

        self.table: DataTable = self.new_table(
            vt_symbols=self.vt_symbols,
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
        self.handler.update_tick(tick)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """Callback of 1-minute candle bars update"""
        # Check breakout and execute trading every minute
        bar: BarData = bars[self.trading_symbol]
        self.check_breakout(bar)
        self.execute_trading(bars, tick_add=5)

        # Update bars into data table
        finished: bool = self.table.update_bars(bars)

        # Calculate indicators every hour
        if finished:
            table_df: pd.DataFrame | None = self.table.get_df()
            if table_df is None:
                return
            df: pd.DataFrame = cast(pd.DataFrame, table_df.xs(self.trading_symbol, level=1))

            close_arr: np.ndarray = df["close_price"].to_numpy()
            sma_arr: np.ndarray = ta.SMA(close_arr, self.boll_window)
            std_arr: np.ndarray = ta.STDDEV(close_arr, self.boll_window)
            boll_up_arr: np.ndarray = sma_arr + std_arr * self.boll_dev
            boll_down_arr: np.ndarray = sma_arr - std_arr * self.boll_dev
            self.boll_up = boll_up_arr[-1]
            self.boll_down = boll_down_arr[-1]

            high_arr: np.ndarray = df["high_price"].to_numpy()
            low_arr: np.ndarray = df["low_price"].to_numpy()
            atr_arr: np.ndarray = ta.ATR(high_arr, low_arr, close_arr, self.atr_window)
            self.atr_value = atr_arr[-1]
            self.trading_size = round(self.risk_level / self.atr_value, 2)

            self.trading_allowed = True

        # Put event to upgrade GUI
        self.put_event()

    def check_breakout(self, bar: BarData) -> None:
        """Check critical level breakout"""
        if not self.trading_allowed:
            return

        # Holding no position
        last_target: float = self.get_target(self.trading_symbol)

        if not last_target:
            if bar.high_price >= self.boll_up:
                self.set_target(self.trading_symbol, self.trading_size)
                self.intra_trade_high = bar.close_price

                self.trading_allowed = False
            elif bar.low_price <= self.boll_down:
                self.set_target(self.trading_symbol, -self.trading_size)
                self.intra_trade_low = bar.close_price

                self.trading_allowed = False
        # Holding long position
        elif last_target > 0:
            if bar.low_price <= self.long_stop:
                self.set_target(self.trading_symbol, 0)

                self.trading_allowed = False
            else:
                self.intra_trade_high = max(self.intra_trade_high, bar.high_price)
                self.long_stop = self.intra_trade_high - self.atr_value * self.trailing_multiplier
        # Holding short position
        elif last_target < 0:
            if bar.high_price >= self.short_stop:
                self.set_target(self.trading_symbol, 0)

                self.trading_allowed = False
            else:
                self.intra_trade_low = min(self.intra_trade_low, bar.low_price)
                self.short_stop = self.intra_trade_low + self.atr_value * self.trailing_multiplier

    def on_trade(self, trade: TradeData) -> None:
        """Callback of trade update"""
        # Get latest pos
        self.trading_pos = self.get_pos(self.trading_symbol)

        # Put event to upgrade GUI
        self.put_event()

    def on_order(self, order: OrderData) -> None:
        """Callback of order update"""
        pass
