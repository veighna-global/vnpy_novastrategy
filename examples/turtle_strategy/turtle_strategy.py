from vnpy_evo.trader.utility import round_to

from vnpy_novastrategy import (
    StrategyTemplate,
    BarData, TickData,
    TradeData, OrderData,
    ArrayManager, BarGenerator,
    Interval, datetime
)


class TurtleStrategy(StrategyTemplate):
    """Classic turtle-trading strategy"""

    author: str = "VeighNa Global"

    entry_window: int = 30
    exit_window: int = 9
    atr_window: int = 14
    risk_level: float = 5000

    trading_size: float = 0.0
    trading_target: float = 0.0
    trading_pos: float = 0.0
    entry_up: float = 0.0
    entry_down: float = 0.0
    exit_up: float = 0.0
    exit_down: float = 0.0
    atr_value: float = 0.0
    long_entry: float = 0.0
    short_entry: float = 0.0
    long_stop: float = 0.0
    short_stop: float = 0.0

    parameters = [
        "entry_window",
        "exit_window",
        "atr_window",
        "risk_level"
    ]
    variables = [
        "trading_size",
        "trading_target",
        "trading_pos",
        "entry_up",
        "entry_down",
        "exit_up",
        "exit_down",
        "trading_size",
        "atr_value"
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

        if not self.am.inited:
            return

        if not self.trading_target:
            self.atr_value = self.am.atr(self.atr_window)
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

        trading_volume: int = self.trading_target - self.trading_pos
        pricetick: float = self.get_pricetick(self.trading_symbol)

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

        self.entry_up, self.entry_down = self.am.donchian(self.entry_window)
        self.exit_up, self.exit_down = self.am.donchian(self.exit_window)

    def on_trade(self, trade: TradeData) -> None:
        """Callback of trade update"""
        self.trading_pos = self.get_pos(self.trading_symbol)

        self.put_event()

    def on_order(self, order: OrderData) -> None:
        """Callback of order update"""
        pass
