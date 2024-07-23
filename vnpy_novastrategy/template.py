from typing import TYPE_CHECKING
from collections import defaultdict

from vnpy_evo.trader.constant import Interval, Direction, Offset
from vnpy_evo.trader.object import BarData, TickData, OrderData, TradeData
from vnpy_evo.trader.utility import virtual

if TYPE_CHECKING:
    from .engine import StrategyEngine


class StrategyTemplate:
    """Strategy template"""

    author: str = ""
    parameters: list = []
    variables: list = []

    def __init__(
        self,
        strategy_engine: "StrategyEngine",
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ) -> None:
        """
        Normally no need to call this __init__ when implementing a strategy.
        """
        self.strategy_engine: "StrategyEngine" = strategy_engine

        self.strategy_name: str = strategy_name
        self.vt_symbols: list[str] = vt_symbols

        # Strategy status variable
        self.inited: bool = False
        self.trading: bool = False

        self.pos_data: dict[str, int] = defaultdict(int)

        self.active_orderids: set[str] = set()

        # Update strategy setting
        self.update_setting(setting)

    def update_setting(self, setting: dict) -> None:
        """Update parameters from setting"""
        for name in self.parameters:
            if name in setting:
                setattr(self, name, setting[name])

    @classmethod
    def get_class_parameters(cls) -> dict:
        """Get strategy default parameters"""
        class_parameters: dict = {}
        for name in cls.parameters:
            class_parameters[name] = getattr(cls, name)
        return class_parameters

    def get_parameters(self) -> dict:
        """Get strategy object parameters"""
        strategy_parameters: dict = {}
        for name in self.parameters:
            strategy_parameters[name] = getattr(self, name)
        return strategy_parameters

    def get_variables(self) -> dict:
        """Get strategy object variables"""
        strategy_variables: dict = {}
        for name in self.variables:
            strategy_variables[name] = getattr(self, name)
        return strategy_variables

    def get_data(self) -> dict:
        """Get strategy data dict"""
        strategy_data: dict = {
            "strategy_name": self.strategy_name,
            "vt_symbols": self.vt_symbols,
            "class_name": self.__class__.__name__,
            "author": self.author,
            "inited": self.inited,
            "trading": self.trading,
            "pos_data": dict(self.pos_data),
            "parameters": self.get_parameters(),
            "variables": self.get_variables(),
        }
        return strategy_data

    @virtual
    def on_init(self) -> None:
        """Callback when strategy is inited"""
        pass

    @virtual
    def on_start(self) -> None:
        """Callback when strategy is started"""
        pass

    @virtual
    def on_stop(self) -> None:
        """Callback when strategy is stoped"""
        pass

    @virtual
    def on_tick(self, tick: TickData) -> None:
        """Callback of tick data update"""
        pass

    @virtual
    def on_bars(self, bars: dict[str, BarData]) -> None:
        """Callback of candle bar update"""
        pass

    @virtual
    def on_trade(self, trade: TradeData) -> None:
        """Callback of trade update"""
        pass

    @virtual
    def on_order(self, order: OrderData) -> None:
        """Callback of order update"""
        pass

    def update_trade(self, trade: TradeData) -> None:
        """Calculate strategy pos data before calling on_trade"""
        if trade.direction == Direction.LONG:
            self.pos_data[trade.vt_symbol] += trade.volume
        else:
            self.pos_data[trade.vt_symbol] -= trade.volume

        self.on_trade(trade)

    def update_order(self, order: OrderData) -> None:
        """Update active orderid set beforce calling on_order"""
        vt_orderid: str = order.vt_orderid

        if not order.is_active() and vt_orderid in self.active_orderids:
            self.active_orderids.remove(vt_orderid)

        self.on_order(order)

    def buy(self, vt_symbol: str, price: float, volume: float) -> str:
        """Send buy order"""
        return self.send_order(vt_symbol, Direction.LONG, Offset.OPEN, price, volume)

    def sell(self, vt_symbol: str, price: float, volume: float) -> str:
        """Send sell order"""
        return self.send_order(vt_symbol, Direction.SHORT, Offset.CLOSE, price, volume)

    def short(self, vt_symbol: str, price: float, volume: float) -> str:
        """Send short order"""
        return self.send_order(vt_symbol, Direction.SHORT, Offset.OPEN, price, volume)

    def cover(self, vt_symbol: str, price: float, volume: float) -> str:
        """Send cover order"""
        return self.send_order(vt_symbol, Direction.LONG, Offset.CLOSE, price, volume)

    def send_order(
        self,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
    ) -> str:
        """Send new order"""
        if not self.trading:
            return ""

        vt_orderid: str = self.strategy_engine.send_order(
            strategy=self,
            vt_symbol=vt_symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume
        )

        self.active_orderids.add(vt_orderid)

        return vt_orderid

    def cancel_order(self, vt_orderid: str) -> None:
        """Cancel existing order"""
        if self.trading:
            self.strategy_engine.cancel_order(self, vt_orderid)

    def cancel_all(self) -> None:
        """Cancel all active orders"""
        for vt_orderid in self.active_orderids:
            self.cancel_order(vt_orderid)

    def get_pos(self, vt_symbol: str) -> int:
        """Get current pos of a contract"""
        return self.pos_data.get(vt_symbol, 0)

    def write_log(self, msg: str) -> None:
        """Write log"""
        self.strategy_engine.write_log(msg, self)

    def get_pricetick(self, vt_symbol: str) -> float:
        """Get pricetick of a contract"""
        return self.strategy_engine.get_pricetick(self, vt_symbol)

    def get_size(self, vt_symbol: str) -> int:
        """Get size of a contract"""
        return self.strategy_engine.get_size(self, vt_symbol)

    def load_bars(self, days: int, interval: Interval) -> None:
        """Load history data to init a strategy"""
        self.strategy_engine.load_bars(self, days, interval)

    def put_event(self) -> None:
        """Put strategy UI update event"""
        if self.inited:
            self.strategy_engine.put_strategy_event(self)

    def sync_data(self) -> None:
        """Sync strategy data into files"""
        if self.trading:
            self.strategy_engine.sync_strategy_data(self)
