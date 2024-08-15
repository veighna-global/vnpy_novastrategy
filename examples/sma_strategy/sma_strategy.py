from vnpy_novastrategy import (
    StrategyTemplate,
    BarData, TickData,
    TradeData, OrderData,
    ArrayManager, BarGenerator,
    Interval, datetime
)


class SmaStrategy(StrategyTemplate):
    """Double SMA (simple moving average) strategy"""

    author: str = "VeighNa Global"

    fast_window: int = 5
    slow_window: int = 20
    trading_symbol: str = ""
    trading_size: int = 1

    fast_ma: int = 0
    slow_ma: int = 0
    trading_target: int = 0
    trading_pos: int = 0

    parameters: list = [
        "fast_window",
        "slow_window",
        "trading_symbol",
        "trading_size",
    ]

    variables: list = [
        "fast_ma",
        "slow_ma",
        "trading_target",
        "trading_pos",
    ]

    def on_init(self) -> None:
        """Callback when strategy is inited"""
        self.trading_symbol: str = self.vt_symbols[0]

        self.bar_dt: datetime = None

        self.bg: BarGenerator = BarGenerator(
            on_bar=lambda bar: None,
            window=1,
            on_window_bar=self.on_window_bar,
            interval=Interval.HOUR
        )

        self.am: ArrayManager = ArrayManager()

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

        bar_dt: datetime = bar.datetime
        if self.bar_dt and bar_dt == self.bar_dt:
            return
        self.bar_dt = bar_dt

        bars: dict = {bar.vt_symbol: bar}
        self.on_bars(bars)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """Callback of 1-minute candle bars update"""
        self.cancel_all()

        bar: BarData = bars[self.trading_symbol]

        self.bg.update_bar(bar)

    def on_window_bar(self, bar: BarData) -> None:
        """Callback of window bar update"""
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        self.fast_ma = self.am.sma(self.fast_window)
        self.slow_ma = self.am.sma(self.slow_window)

        if self.fast_ma > self.slow_ma:
            self.trading_target = self.trading_size
        else:
            self.trading_target = -self.trading_size

        trading_volume: int = self.trading_target - self.trading_pos

        if trading_volume > 0:
            self.buy(self.trading_symbol, bar.close_price * 1.01, abs(trading_volume))
        else:
            self.short(self.trading_symbol, bar.close_price * 0.99, abs(trading_volume))

        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        """Callback of trade update"""
        self.trading_pos = self.get_pos(self.trading_symbol)

        self.put_event()

    def on_order(self, order: OrderData) -> None:
        """Callback of order update"""
        pass
