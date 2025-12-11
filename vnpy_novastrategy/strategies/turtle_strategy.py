from typing import cast

from vnpy_novastrategy import (
    StrategyTemplate,
    BarData, TickData,
    TradeData, OrderData,
    ArrayManager, BarGenerator,
    Interval, datetime,
    Parameter, Variable, round_to
)


class TurtleStrategy(StrategyTemplate):
    """Classic turtle-trading strategy"""

    author: str = "VeighNa Global"

    entry_window: Parameter[int] = Parameter(30)
    exit_window: Parameter[int] = Parameter(9)
    atr_window: Parameter[int] = Parameter(14)
    risk_level: Parameter[float] = Parameter(5000.0)

    trading_size: Variable[float] = Variable(0.0)
    trading_target: Variable[float] = Variable(0.0)
    trading_pos: Variable[float] = Variable(0.0)
    entry_up: Variable[float] = Variable(0.0)
    entry_down: Variable[float] = Variable(0.0)
    exit_up: Variable[float] = Variable(0.0)
    exit_down: Variable[float] = Variable(0.0)
    atr_value: Variable[float] = Variable(0.0)
    long_entry: Variable[float] = Variable(0.0)
    short_entry: Variable[float] = Variable(0.0)
    long_stop: Variable[float] = Variable(0.0)
    short_stop: Variable[float] = Variable(0.0)

    def on_init(self) -> None:
        """Callback when strategy is inited"""
        self.trading_symbol: str = self.vt_symbols[0]

        self.bar_dt: datetime | None = None

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
        if not tick.extra:
            return

        bar: BarData | None = tick.extra.get("bar", None)
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

        if not self.am.inited:
            return

        if not self.trading_target:
            self.atr_value = cast(float, self.am.atr(self.atr_window))
            self.trading_size = round(self.risk_level / self.atr_value, 2)

            if bar.high_price >= self.entry_up:
                self.trading_target = self.trading_size
            elif bar.low_price <= self.entry_down:
                self.trading_target = -self.trading_size
        elif self.trading_target > 0:
            if bar.low_price <= self.exit_down:
                self.trading_target = 0
        elif self.trading_target < 0:
            if bar.high_price >= self.exit_up:
                self.trading_target = 0

        trading_volume: float = self.trading_target - self.trading_pos
        pricetick: float = cast(float, self.get_pricetick(self.trading_symbol))

        if trading_volume > 0:
            buy_price: float = round_to(bar.close_price * 1.01, pricetick)
            self.buy(self.trading_symbol, buy_price, abs(trading_volume))
        elif trading_volume < 0:
            short_price: float = round_to(bar.close_price * 0.99, pricetick)
            self.short(self.trading_symbol, short_price, abs(trading_volume))

        self.put_event()

    def on_window_bar(self, bar: BarData) -> None:
        """Callback of window bar update"""
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        entry_result = self.am.donchian(self.entry_window)
        self.entry_up = float(entry_result[0])
        self.entry_down = float(entry_result[1])

        exit_result = self.am.donchian(self.exit_window)
        self.exit_up = float(exit_result[0])
        self.exit_down = float(exit_result[1])

    def on_trade(self, trade: TradeData) -> None:
        """Callback of trade update"""
        self.trading_pos = self.get_pos(self.trading_symbol)

        self.put_event()

    def on_order(self, order: OrderData) -> None:
        """Callback of order update"""
        pass
