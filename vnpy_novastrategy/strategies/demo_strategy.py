from vnpy_novastrategy import (
    StrategyTemplate,
    BarData, TickData,
    TradeData, OrderData
)


class DemoStrategy(StrategyTemplate):
    """Simple strategy demo"""

    author: str = "VeighNa Global"

    fast_window: int = 5
    slow_window: int = 20

    fast_ma: int = 0
    slow_ma: int = 0

    parameters: list = [
        "fast_window",
        "slow_window"
    ]

    variables: list = [
        "fast_ma",
        "slow_ma"
    ]

    def on_init(self) -> None:
        """Callback when strategy is inited"""
        self.write_log("Strategy is inited.")

    def on_start(self) -> None:
        """Callback when strategy is started"""
        self.write_log("Strategy is started.")

    def on_stop(self) -> None:
        """Callback when strategy is stoped"""
        self.write_log("Strategy is stopped.")

    def on_tick(self, tick: TickData) -> None:
        """Callback of tick data update"""
        pass

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """Callback of candle bar update"""
        pass

    def on_trade(self, trade: TradeData) -> None:
        """Callback of trade update"""
        pass

    def on_order(self, order: OrderData) -> None:
        """Callback of order update"""
        pass
