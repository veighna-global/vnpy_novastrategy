import importlib
import glob
import traceback
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Type, Callable, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from vnpy.event import Event, EventEngine
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.object import (
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
    HistoryRequest,
    LogData,
    TickData,
    OrderData,
    TradeData,
    BarData,
    ContractData
)
from vnpy.trader.event import (
    EVENT_TICK,
    EVENT_ORDER,
    EVENT_TRADE
)
from vnpy.trader.constant import (
    Direction,
    OrderType,
    Interval,
    Offset
)
from vnpy.trader.utility import load_json, save_json, extract_vt_symbol, round_to
from vnpy.trader.database import BaseDatabase, get_database, DB_TZ

from .base import (
    APP_NAME,
    EVENT_NOVA_LOG,
    EVENT_NOVA_STRATEGY
)
from .template import StrategyTemplate
from .table import LiveDataTable


class StrategyEngine(BaseEngine):
    """Nova strategy engine"""

    setting_filename: str = "nova_strategy_setting.json"
    data_filename: str = "nova_strategy_data.json"

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__(main_engine, event_engine, APP_NAME)

        self.strategy_data: dict[str, dict] = {}

        self.classes: dict[str, Type[StrategyTemplate]] = {}
        self.strategies: dict[str, StrategyTemplate] = {}

        self.symbol_strategy_map: dict[str, list[StrategyTemplate]] = defaultdict(list)
        self.orderid_strategy_map: dict[str, StrategyTemplate] = {}

        self.init_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

        self.vt_tradeids: set[str] = set()

        self.database: BaseDatabase = get_database()

    def init_engine(self) -> None:
        """Initialize strategy engine"""
        self._load_strategy_class()
        self._load_strategy_setting()
        self._load_strategy_data()
        self._register_event()
        self.write_log("Nova strategy engine is initialized.")

    def close(self) -> None:
        """Close strategy engine"""
        self.stop_all_strategies()

    def _register_event(self) -> None:
        """Register event handler"""
        self.event_engine.register(EVENT_TICK, self._process_tick_event)
        self.event_engine.register(EVENT_ORDER, self._process_order_event)
        self.event_engine.register(EVENT_TRADE, self._process_trade_event)

    def _process_tick_event(self, event: Event) -> None:
        """Process tick data event"""
        tick: TickData = event.data

        strategies: list = self.symbol_strategy_map[tick.vt_symbol]
        if not strategies:
            return

        for strategy in strategies:
            if strategy.inited:
                self._call_strategy_func(strategy, strategy.on_tick, tick)

    def _process_order_event(self, event: Event) -> None:
        """Process order data event"""
        order: OrderData = event.data

        strategy: Optional[StrategyTemplate] = self.orderid_strategy_map.get(order.vt_orderid, None)
        if not strategy:
            return

        self._call_strategy_func(strategy, strategy.update_order, order)

    def _process_trade_event(self, event: Event) -> None:
        """Process trade data event"""
        trade: TradeData = event.data

        # Filter duplicate data
        if trade.vt_tradeid in self.vt_tradeids:
            return
        self.vt_tradeids.add(trade.vt_tradeid)

        strategy: Optional[StrategyTemplate] = self.orderid_strategy_map.get(trade.vt_orderid, None)
        if not strategy:
            return

        self._call_strategy_func(strategy, strategy.update_trade, trade)

    def subscribe_data(self, strategy: StrategyTemplate, vt_symbol: str) -> bool:
        """Subscribe new data"""
        contract = self.main_engine.get_contract(vt_symbol)
        if not contract:
            return False

        req = SubscribeRequest(contract.symbol, contract.exchange)
        self.main_engine.subscribe(req, contract.gateway_name)

        self.symbol_strategy_map[vt_symbol].append(strategy)

        return True

    def send_order(
        self,
        strategy: StrategyTemplate,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
    ) -> str:
        """Send new order"""
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)
        if not contract:
            self.write_log(f"Failed to send order, contract not found: {vt_symbol}", strategy)
            return ""

        price: float = round_to(price, contract.pricetick)
        volume: float = round_to(volume, contract.min_volume)

        req: OrderRequest = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            offset=offset,
            type=OrderType.LIMIT,
            price=price,
            volume=volume,
            reference=f"{APP_NAME}_{strategy.strategy_name}"
        )

        vt_orderid: str = self.main_engine.send_order(req, contract.gateway_name)
        self.orderid_strategy_map[vt_orderid] = strategy

        return vt_orderid

    def cancel_order(self, strategy: StrategyTemplate, vt_orderid: str) -> None:
        """Cancel existing order"""
        order: Optional[OrderData] = self.main_engine.get_order(vt_orderid)
        if not order:
            self.write_log(f"Failed to cancel order, order not found: {vt_orderid}", strategy)
            return

        req: CancelRequest = order.create_cancel_request()
        self.main_engine.cancel_order(req, order.gateway_name)

    def get_pricetick(self, strategy: StrategyTemplate, vt_symbol: str) -> float:
        """Get contract pricetick"""
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)

        if contract:
            return contract.pricetick
        else:
            return None

    def get_size(self, strategy: StrategyTemplate, vt_symbol: str) -> int:
        """Get contract size"""
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)

        if contract:
            return contract.size
        else:
            return None

    def get_min_volume(self, strategy: StrategyTemplate, vt_symbol: str) -> float:
        """Get min volume of a contract"""
        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)

        if contract:
            return contract.min_volume
        else:
            return None

    def new_table(
        self,
        vt_symbols: list[str],
        size: int,
        window: int,
        interval: Interval,
        extra_fields: list[str]
    ) -> LiveDataTable:
        """Create a new DataTable"""
        return LiveDataTable(
            vt_symbols=vt_symbols,
            size=size,
            window=window,
            interval=interval,
            extra_fields=extra_fields,
        )

    def load_bars(self, strategy: StrategyTemplate, days: int, interval: Interval) -> None:
        """Load history bar data for portfolio"""
        vt_symbols: list = strategy.vt_symbols
        dts: set[datetime] = set()
        history_data: dict[str, dict] = defaultdict(dict)

        # Load bar data from gateway
        for vt_symbol in vt_symbols:
            data: list[BarData] = self._load_bar(vt_symbol, days, interval)

            for bar in data:
                bars: dict[str, BarData] = history_data.setdefault(bar.datetime, {})
                bars[bar.vt_symbol] = bar

        dts: list = list(history_data.keys())
        dts.sort()

        for dt in dts:
            bars = history_data[dt]
            self._call_strategy_func(strategy, strategy.on_bars, bars)

    def _load_bar(self, vt_symbol: str, days: int, interval: Interval) -> list[BarData]:
        """Load history bar data for specific symbol"""
        symbol, exchange = extract_vt_symbol(vt_symbol)

        end: datetime = datetime.now(DB_TZ)
        start: datetime = end - timedelta(days)

        contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)

        req: HistoryRequest = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end
        )
        data: list[BarData] = self.main_engine.query_history(req, contract.gateway_name)

        return data

    def _call_strategy_func(self, strategy: StrategyTemplate, func: Callable, params: object = None) -> None:
        """Call strategy method with exception process"""
        try:
            if params:
                func(params)
            else:
                func()
        except Exception:
            strategy.trading = False
            strategy.inited = False

            msg: str = f"Strategy stopped due to exception\n{traceback.format_exc()}"
            self.write_log(msg, strategy)

    def add_strategy(
        self,
        class_name: str,
        strategy_name: str,
        vt_symbols: list,
        setting: dict
    ) -> None:
        """Add a strategy instance"""
        if strategy_name in self.strategies:
            self.write_log(f"Add strategy failed, name already exists: {strategy_name}.")
            return

        strategy_class: Optional[StrategyTemplate] = self.classes.get(class_name, None)
        if not strategy_class:
            self.write_log(f"Add strategy failed, strategy class not found: {class_name}.")
            return

        strategy: StrategyTemplate = strategy_class(self, strategy_name, vt_symbols, setting)
        self.strategies[strategy_name] = strategy

        for vt_symbol in vt_symbols:
            strategies: list = self.symbol_strategy_map[vt_symbol]
            strategies.append(strategy)

        self._save_strategy_setting()
        self.put_strategy_event(strategy)

    def init_strategy(self, strategy_name: str) -> None:
        """Submit a initialization task"""
        self.init_executor.submit(self._init_strategy, strategy_name)

    def _init_strategy(self, strategy_name: str) -> None:
        """Initialize a strategy"""
        strategy: StrategyTemplate = self.strategies[strategy_name]

        if strategy.inited:
            self.write_log(f"Initialize strategy failed, {strategy_name} already initialized.")
            return

        self.write_log(f"Initilizing strategy {strategy_name}.")

        # Call strategy.on_init method
        self._call_strategy_func(strategy, strategy.on_init)

        # Restore strategy variables
        data: Optional[dict] = self.strategy_data.get(strategy_name, None)
        if data:
            pos_data: dict = data.get("pos_data", {})
            strategy.pos_data.update(pos_data)

            variables: dict = data.get("variables", {})
            for name in strategy.variables:
                value: Optional[object] = variables.get(name, None)
                if value is None:
                    continue
                setattr(strategy, name, value)

        # Subscribe market data
        for vt_symbol in strategy.vt_symbols:
            contract: Optional[ContractData] = self.main_engine.get_contract(vt_symbol)
            if contract:
                req: SubscribeRequest = SubscribeRequest(symbol=contract.symbol, exchange=contract.exchange)
                self.main_engine.subscribe(req, contract.gateway_name)
            else:
                self.write_log(f"Subscribe market data failed, contract not found: {vt_symbol}", strategy)

        # Put strategy event to notify
        strategy.inited = True
        self.put_strategy_event(strategy)
        self.write_log(f"Initialization of {strategy_name} completed.")

    def start_strategy(self, strategy_name: str) -> None:
        """Start a strategy"""
        strategy: StrategyTemplate = self.strategies[strategy_name]
        if not strategy.inited:
            self.write_log(f"Start strategy failed, {strategy.strategy_name} is not initilized yet.")
            return

        if strategy.trading:
            self.write_log(f"Start strategy failed, {strategy_name} is already started.")
            return

        # Call strategy.on_start method
        self._call_strategy_func(strategy, strategy.on_start)

        # Put strategy event to notify
        strategy.trading = True
        self.put_strategy_event(strategy)

    def stop_strategy(self, strategy_name: str) -> None:
        """Stop a strategy"""
        strategy: StrategyTemplate = self.strategies[strategy_name]
        if not strategy.trading:
            return

        # Call strategy.on_stop metho
        self._call_strategy_func(strategy, strategy.on_stop)

        # Set trading status to False
        strategy.trading = False

        # Cancel all active orders of the strategy
        strategy.cancel_all()

        # Sync strategy data to file
        self.sync_strategy_data(strategy)

        # Put strategy event to notify
        self.put_strategy_event(strategy)

    def edit_strategy(self, strategy_name: str, setting: dict) -> None:
        """Edit strategy parameters"""
        strategy: StrategyTemplate = self.strategies[strategy_name]
        strategy.update_setting(setting)

        self._save_strategy_setting()
        self.put_strategy_event(strategy)

    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy instance"""
        strategy: StrategyTemplate = self.strategies[strategy_name]
        if strategy.trading:
            self.write_log(f"Remove strategy failed, please stop {strategy.strategy_name} first.")
            return

        for vt_symbol in strategy.vt_symbols:
            strategies: list = self.symbol_strategy_map[vt_symbol]
            strategies.remove(strategy)

        for vt_orderid in strategy.active_orderids:
            if vt_orderid in self.orderid_strategy_map:
                self.orderid_strategy_map.pop(vt_orderid)

        self.strategies.pop(strategy_name)
        self._save_strategy_setting()

        self.strategy_data.pop(strategy_name, None)
        save_json(self.data_filename, self.strategy_data)

        return True

    def _load_strategy_class(self) -> None:
        """Load strategy class from files"""
        path1: Path = Path(__file__).parent.joinpath("strategies")
        self._load_strategy_class_from_folder(path1, "vnpy_novastrategy.strategies")

        path2: Path = Path.cwd().joinpath("strategies")
        self._load_strategy_class_from_folder(path2, "strategies")

    def _load_strategy_class_from_folder(self, path: Path, module_name: str = "") -> None:
        """Load strategy class from specific folder"""
        for suffix in ["py", "pyd", "so"]:
            pathname: str = str(path.joinpath(f"*.{suffix}"))
            for filepath in glob.glob(pathname):
                stem: str = Path(filepath).stem
                strategy_module_name: str = f"{module_name}.{stem}"
                self._load_strategy_class_from_module(strategy_module_name)

    def _load_strategy_class_from_module(self, module_name: str) -> None:
        """Load strategy class from specific file"""
        try:
            module: ModuleType = importlib.import_module(module_name)

            for name in dir(module):
                value = getattr(module, name)
                if (isinstance(value, type) and issubclass(value, StrategyTemplate) and value is not StrategyTemplate):
                    self.classes[value.__name__] = value
        except Exception:
            msg: str = f"Load strategy file {module_name} failed due to exception: \n{traceback.format_exc()}"
            self.write_log(msg)

    def _load_strategy_data(self) -> None:
        """Load strategy data from data file"""
        self.strategy_data = load_json(self.data_filename)

    def sync_strategy_data(self, strategy: StrategyTemplate) -> None:
        """Sync strategy data to data file"""
        data: dict = {
            "variables": strategy.get_variables(),
            "pos_data": dict(strategy.pos_data)
        }

        self.strategy_data[strategy.strategy_name] = data
        save_json(self.data_filename, self.strategy_data)

    def get_all_strategy_class_names(self) -> list:
        """Get all available strategy names"""
        return list(self.classes.keys())

    def get_strategy_class_parameters(self, class_name: str) -> dict:
        """Get default parameters of a strategy class"""
        strategy_class: StrategyTemplate = self.classes[class_name]

        parameters: dict = {}
        for name in strategy_class.parameters:
            parameters[name] = getattr(strategy_class, name)

        return parameters

    def get_strategy_parameters(self, strategy_name) -> dict:
        """Get parameters of a strategy instance"""
        strategy: StrategyTemplate = self.strategies[strategy_name]
        return strategy.get_parameters()

    def init_all_strategies(self) -> None:
        """Initialize all strategy instances"""
        for strategy_name in self.strategies.keys():
            self.init_strategy(strategy_name)

    def start_all_strategies(self) -> None:
        """Start all strategy instances"""
        for strategy_name in self.strategies.keys():
            self.start_strategy(strategy_name)

    def stop_all_strategies(self) -> None:
        """Stop all strategy instances"""
        for strategy_name in self.strategies.keys():
            self.stop_strategy(strategy_name)

    def _load_strategy_setting(self) -> None:
        """Load strategy setting from file"""
        strategy_setting: dict = load_json(self.setting_filename)

        for strategy_name, strategy_config in strategy_setting.items():
            self.add_strategy(
                strategy_config["class_name"],
                strategy_name,
                strategy_config["vt_symbols"],
                strategy_config["setting"]
            )

    def _save_strategy_setting(self) -> None:
        """Save strategy setting to file"""
        strategy_setting: dict = {}

        for name, strategy in self.strategies.items():
            strategy_setting[name] = {
                "class_name": strategy.__class__.__name__,
                "vt_symbols": strategy.vt_symbols,
                "setting": strategy.get_parameters()
            }

        save_json(self.setting_filename, strategy_setting)

    def put_strategy_event(self, strategy: StrategyTemplate) -> None:
        """Put event to update UI"""
        data: dict = strategy.get_data()
        event: Event = Event(EVENT_NOVA_STRATEGY, data)
        self.event_engine.put(event)

    def write_log(self, msg: str, strategy: StrategyTemplate = None) -> None:
        """Write log message"""
        if strategy:
            msg: str = f"{strategy.strategy_name}: {msg}"

        log: LogData = LogData(msg=msg, gateway_name=APP_NAME)
        event: Event = Event(type=EVENT_NOVA_LOG, data=log)
        self.event_engine.put(event)
