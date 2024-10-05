from typing import Callable
from datetime import datetime

from vnpy_evo.trader.object import BarData, TickData


class TickHandler:
    """Handles tick data to generate bars snapshot"""

    def __init__(
        self,
        vt_symbols: list[str],
        on_bars: Callable[[dict[str, BarData]], None]
    ) -> None:
        """"""
        self.vt_symbols: list[str] = vt_symbols
        self.symbol_count: int = len(self.vt_symbols)
        self.on_bars: Callable[[dict[str, BarData]], None] = on_bars

        self.last_bars: dict[str, BarData] = {}
        self.closed_dt: datetime = None

    def update_tick(self, tick: TickData) -> None:
        """Update new tick data"""
        bar: BarData = tick.extra.get("bar", None)
        if not bar:
            return

        if self.closed_dt and bar.datetime < self.closed_dt:
            return

        self.last_bars[bar.vt_symbol] = bar

        if len(self.last_bars) == self.symbol_count:
            self.on_bars(self.last_bars)
            self.last_bars = {}

        self.closed_dt = bar.datetime
