from typing import Union, Type, TYPE_CHECKING
from collections import defaultdict

from vnpy_evo.trader.constant import Interval, Direction, Offset
from vnpy_evo.trader.object import BarData, TickData, OrderData, TradeData
from vnpy_evo.trader.utility import virtual, round_to

from .table import DataTable

if TYPE_CHECKING:
    from .engine import StrategyEngine


class StrategyTemplate:
    """Strategy template"""

    author: str = ""

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
        self.target_data: dict[str, int] = defaultdict(int)

        self.active_orderids: set[str] = set()

        # Initialize parameters and variables lists
        if not hasattr(self, "parameters"):
            self.parameters: list[str] = []

        if not hasattr(self, "variables"):
            self.variables: list[str] = []

        self.variables = ["inited", "trading"] + self.variables

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
        for vt_orderid in list(self.active_orderids):
            self.cancel_order(vt_orderid)

    def execute_trading(self, bars: dict[str, BarData], tick_add: int) -> None:
        """Execute trading according to the difference between target and pos"""
        # Cancel all existing orders
        self.cancel_all()

        # Send new order according to the difference between target and pos
        for vt_symbol, target in self.target_data.items():
            bar: BarData = bars.get(vt_symbol, None)
            if not bar:
                continue

            min_volume: float = self.get_min_volume(vt_symbol)
            pos: float = self.get_pos(vt_symbol)

            trading_volume: int = round_to(target - pos, min_volume)
            if not trading_volume:
                continue

            pricetick: float = self.get_pricetick(vt_symbol)

            if trading_volume > 0:
                buy_price: float = round_to(bar.close_price + pricetick * tick_add, pricetick)
                self.buy(vt_symbol, buy_price, abs(trading_volume))
            elif trading_volume < 0:
                short_price: float = round_to(bar.close_price - pricetick * tick_add, pricetick)
                self.short(vt_symbol, short_price, abs(trading_volume))

    def get_pos(self, vt_symbol: str) -> int:
        """Get current pos of a contract"""
        return self.pos_data.get(vt_symbol, 0)

    def set_target(self, vt_symbol: str, target: int) -> None:
        """Set target pos"""
        self.target_data[vt_symbol] = target

    def get_target(self, vt_symbol: str) -> int:
        """Get target pos"""
        return self.target_data.get(vt_symbol, 0)

    def write_log(self, msg: str) -> None:
        """Write log"""
        self.strategy_engine.write_log(msg, self)

    def get_pricetick(self, vt_symbol: str) -> float:
        """Get pricetick of a contract"""
        return self.strategy_engine.get_pricetick(self, vt_symbol)

    def get_size(self, vt_symbol: str) -> int:
        """Get size of a contract"""
        return self.strategy_engine.get_size(self, vt_symbol)

    def get_min_volume(self, vt_symbol: str) -> float:
        """Get min volume of a contract"""
        return self.strategy_engine.get_min_volume(self, vt_symbol)

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

    def new_table(
        self,
        vt_symbols: list[str],
        size: int = 100,
        window: int = 1,
        interval: Interval = Interval.MINUTE,
        extra_fields: list[str] = None
    ) -> DataTable:
        """Create a new DataTable"""
        return self.strategy_engine.new_table(
            vt_symbols=vt_symbols,
            size=size,
            window=window,
            interval=interval,
            extra_fields=extra_fields,
        )

    def subscribe_data(self, vt_symbol: str) -> bool:
        """Subscribe new data"""
        if vt_symbol in self.vt_symbols:
            return True

        result: bool = self.strategy_engine.subscribe_data(self, vt_symbol)
        if result:
            self.vt_symbols.add(vt_symbol)

        return result


FieldValue = Union[str, int, float, bool]


class StrategyField:
    """Member value of strategy class"""

    def __init__(
        self,
        value: FieldValue,
        type: str,
    ) -> None:
        """"""
        self.value: FieldValue = value
        self.type: str = type

    def __set_name__(self, owner: Type[StrategyTemplate], name: str) -> None:
        """Add field name into related list"""
        if hasattr(owner, self.type):
            names: list[str] = getattr(owner, self.type)
        else:
            names: list[str] = []
            setattr(owner, self.type, names)

        names.append(name)

        setattr(owner, name, self.value)


class Parameter(StrategyField):
    """Strategy parameter member"""

    def __init__(self, value: FieldValue) -> None:
        """"""
        super().__init__(value, "parameters")


class Variable(StrategyField):
    """Strategy variable member"""

    def __init__(self, value: FieldValue) -> None:
        """"""
        super().__init__(value, "variables")
