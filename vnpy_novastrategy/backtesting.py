from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Optional
from functools import lru_cache, partial
from pathlib import Path
import traceback
import pickle

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas import DataFrame
from tqdm import tqdm

from vnpy_evo.trader.constant import Direction, Offset, Interval, Status
from vnpy_evo.trader.database import get_database, BaseDatabase
from vnpy_evo.trader.object import OrderData, TradeData, BarData
from vnpy_evo.trader.utility import round_to, extract_vt_symbol, get_file_path
from vnpy_evo.trader.optimize import (
    OptimizationSetting,
    check_optimization_setting,
    run_bf_optimization,
    run_ga_optimization
)

from .template import StrategyTemplate


INTERVAL_DELTA_MAP: dict[Interval, timedelta] = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}


class BacktestingEngine:
    """
    Supports historical backtesting and parameter optimization
    """

    gateway_name: str = "BACKTESTING"

    def __init__(self) -> None:
        """"""
        self.interval: Interval = None
        self.start: datetime = None
        self.end: datetime = None
        self.capital: float = 1_000_000
        self.risk_free: float = 0
        self.annual_days: int = 240

        self.vt_symbols: set[str] = set()
        self.priceticks: dict[str, float] = {}
        self.sizes: dict[str, float] = {}
        self.rates: dict[str, float] = {}
        self.slippages: dict[str, float] = {}

        self.strategy_class: StrategyTemplate = None
        self.strategy: StrategyTemplate = None
        self.bars: dict[str, BarData] = {}
        self.datetime: datetime = None

        self.days: int = 0
        self.history_data: dict[str, dict] = defaultdict(dict)

        self.limit_order_count: int = 0
        self.limit_orders: dict[str, OrderData] = {}
        self.active_limit_orders: dict[str, OrderData] = {}

        self.trade_count: int = 0
        self.trades: dict[str, TradeData] = {}

        self.logs: list = []

        self.daily_results: dict[date, PortfolioDailyResult] = {}
        self.daily_df: DataFrame = None

    def set_parameters(
        self,
        interval: Interval,
        start: datetime,
        end: datetime,
        capital: int,
        risk_free: float = 0,
        annual_days: int = 365
    ) -> None:
        """Set backtesting parameters"""
        self.interval = interval
        self.start = start
        self.end = end
        self.capital = capital

        self.risk_free = risk_free
        self.annual_days = annual_days

    def add_contract(
        self,
        vt_symbol: str,
        pricetick: float,
        size: float,
        rate: float,
        slippage: float
    ) -> None:
        """Add contract for backtesting"""
        self.vt_symbols.add(vt_symbol)
        self.priceticks[vt_symbol] = pricetick
        self.sizes[vt_symbol] = size
        self.rates[vt_symbol] = rate
        self.slippages[vt_symbol] = slippage

    def add_strategy(self, strategy_class: type, setting: dict) -> None:
        """Add the strategy for backtesting"""
        self.strategy_class = strategy_class

        self.strategy = strategy_class(
            self,
            strategy_class.__name__,
            list(self.vt_symbols),
            setting
        )

    def load_data(self) -> None:
        """Load history data"""
        self.output("Loading history data.")

        if self.start >= self.end:
            self.output("The start time must be earlier than the end time!")
            return

        # Load history data of all symbols
        for vt_symbol in self.vt_symbols:
            data: list[BarData] = load_bar_data(
                vt_symbol,
                self.interval,
                self.start,
                self.end
            )

            for bar in data:
                bars: dict[str, BarData] = self.history_data.setdefault(bar.datetime, {})
                bars[bar.vt_symbol] = bar

            self.output(f"Bar data of {vt_symbol} loaded, total count: {len(data)}.")

        self.output("History data all loaded.")

    def run_backtesting(self, disable_tqdm: bool = False) -> None:
        """Start backtesting"""
        self.strategy.on_init()

        dts: list = list(self.history_data.keys())
        dts.sort()

        # Initialize the strategy with the head part of data
        day_count: int = 0
        ix: int = 0

        for ix, dt in enumerate(dts):
            if self.datetime and dt.day != self.datetime.day:
                day_count += 1
                if day_count >= self.days:
                    break

            try:
                self._new_bars(dt)
            except Exception:
                self.output("Backtesting is finished due to exception!")
                self.output(traceback.format_exc())
                return

        self.strategy.inited = True
        self.output("The strategy is inited.")

        self.strategy.on_start()
        self.strategy.trading = True
        self.output("Starting to replay history data.")

        # Use the data left for replaying
        backtesting_dts: list[datetime] = dts[ix:]

        for dt in tqdm(backtesting_dts, total=len(backtesting_dts), disable=disable_tqdm):
            try:
                self._new_bars(dt)
            except Exception:
                self.output("Backtesting is finished due to exception!")
                self.output(traceback.format_exc())
                return

        self.output("Replaying history data finished.")

    def calculate_result(self) -> DataFrame:
        """Calculate daily marking-to-market PnL"""
        self.output("Calculating daily PnL.")

        if not self.trades:
            self.output("Calculation failed due to empty trade result.")
            return

        for trade in self.trades.values():
            d: date = trade.datetime.date()
            daily_result: PortfolioDailyResult = self.daily_results[d]
            daily_result.add_trade(trade)

        pre_closes: dict = {}
        start_poses: dict = {}

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_closes,
                start_poses,
                self.sizes,
                self.rates,
                self.slippages,
            )

            pre_closes = daily_result.close_prices
            start_poses = daily_result.end_poses

        results: dict = defaultdict(list)

        for daily_result in self.daily_results.values():
            fields: list = [
                "date", "trade_count", "turnover",
                "commission", "slippage", "trading_pnl",
                "holding_pnl", "total_pnl", "net_pnl"
            ]
            for key in fields:
                value = getattr(daily_result, key)
                results[key].append(value)

        if results:
            self.daily_df: DataFrame = DataFrame.from_dict(results).set_index("date")

        self.output("Calculation of daily PnL finished.")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True) -> dict:
        """Calculate strategy performance statistics"""
        self.output("Calculating performance statistics.")

        if df is None:
            df: DataFrame = self.daily_df

        # Initialize statistics
        start_date: str = ""
        end_date: str = ""
        total_days: int = 0
        profit_days: int = 0
        loss_days: int = 0
        end_balance: float = 0
        max_drawdown: float = 0
        max_ddpercent: float = 0
        max_drawdown_duration: int = 0
        total_net_pnl: float = 0
        daily_net_pnl: float = 0
        total_commission: float = 0
        daily_commission: float = 0
        total_slippage: float = 0
        daily_slippage: float = 0
        total_turnover: float = 0
        daily_turnover: float = 0
        total_trade_count: int = 0
        daily_trade_count: int = 0
        total_return: float = 0
        annual_return: float = 0
        daily_return: float = 0
        return_std: float = 0
        sharpe_ratio: float = 0
        return_drawdown_ratio: float = 0

        positive_balance: bool = False

        # Calculate balance related statistics
        if df is not None:
            df["balance"] = df["net_pnl"].cumsum() + self.capital
            df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)
            df["highlevel"] = df["balance"].rolling(min_periods=1, window=len(df), center=False).max()
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # Check if margin call is raised
            positive_balance = (df["balance"] > 0).all()
            if not positive_balance:
                self.output("Calculation failed due to margin call during backtesting!")

        # Calculate statistics
        if positive_balance:
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days: int = len(df)
            profit_days: int = len(df[df["net_pnl"] > 0])
            loss_days: int = len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            max_drawdown_end = df["drawdown"].idxmin()

            if isinstance(max_drawdown_end, date):
                max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()
                max_drawdown_duration: int = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration: int = 0

            total_net_pnl: float = df["net_pnl"].sum()
            daily_net_pnl: float = total_net_pnl / total_days

            total_commission: float = df["commission"].sum()
            daily_commission: float = total_commission / total_days

            total_slippage: float = df["slippage"].sum()
            daily_slippage: float = total_slippage / total_days

            total_turnover: float = df["turnover"].sum()
            daily_turnover: float = total_turnover / total_days

            total_trade_count: int = df["trade_count"].sum()
            daily_trade_count: int = total_trade_count / total_days

            total_return: float = (end_balance / self.capital - 1) * 100
            annual_return: float = total_return / total_days * self.annual_days
            daily_return: float = df["return"].mean() * 100
            return_std: float = df["return"].std() * 100

            if return_std:
                daily_risk_free: float = self.risk_free / np.sqrt(self.annual_days)
                sharpe_ratio: float = (daily_return - daily_risk_free) / return_std * np.sqrt(self.annual_days)
            else:
                sharpe_ratio: float = 0

            return_drawdown_ratio: float = -total_net_pnl / max_drawdown

        # Output result
        if output:
            self.output("-" * 30)
            self.output(f"Start Date:\t{start_date}")
            self.output(f"End Date:\t{end_date}")

            self.output(f"Total Days:\t{total_days}")
            self.output(f"Profit Days:\t{profit_days}")
            self.output(f"Loss Days:\t{loss_days}")

            self.output(f"Start Balance:\t{self.capital:,.2f}")
            self.output(f"End Balance:\t{end_balance:,.2f}")

            self.output(f"Total Return:\t{total_return:,.2f}%")
            self.output(f"Annual Return:\t{annual_return:,.2f}%")
            self.output(f"Max Drawdown: \t{max_drawdown:,.2f}")
            self.output(f"Max Drawdown(%): {max_ddpercent:,.2f}%")
            self.output(f"Max Drawdown Duration: \t{max_drawdown_duration}")

            self.output(f"Total PnL:\t{total_net_pnl:,.2f}")
            self.output(f"Total Commission:\t{total_commission:,.2f}")
            self.output(f"Total Slippage:\t{total_slippage:,.2f}")
            self.output(f"Total Turnover:\t{total_turnover:,.2f}")
            self.output(f"Total Trades:\t{total_trade_count}")

            self.output(f"Daily PnL:\t{daily_net_pnl:,.2f}")
            self.output(f"Daily Commission:\t{daily_commission:,.2f}")
            self.output(f"Daily Slippage:\t{daily_slippage:,.2f}")
            self.output(f"Daily Turnover:\t{daily_turnover:,.2f}")
            self.output(f"Daily Trades:\t{daily_trade_count}")

            self.output(f"Daily Return:\t{daily_return:,.2f}%")
            self.output(f"Return Std:\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio:\t{sharpe_ratio:,.2f}")
            self.output(f"Return Drawdown Ratio:\t{return_drawdown_ratio:,.2f}")

        statistics: dict = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        # Filter infinite values
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output("Calculation of performance statistics finished.")
        return statistics

    def show_chart(self, df: DataFrame = None) -> None:
        """Show strategy performance chart"""
        if df is None:
            df: DataFrame = self.daily_df

        if df is None:
            return

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Daily PnL", "PnL Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df.index,
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily PnL")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def run_bf_optimization(
        self,
        optimization_setting: OptimizationSetting,
        output=True,
        max_workers: int = None
    ) -> list:
        """Run brutal force optimization"""
        if not check_optimization_setting(optimization_setting):
            return

        temp_path: Path = get_file_path("history_temp")
        with open(temp_path, mode="wb") as f:
            pickle.dump(self.history_data, f)

        evaluate_func: callable = wrap_evaluate(self, optimization_setting.target_name)
        results: list = run_bf_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            output=self.output,
        )

        temp_path.unlink()

        if output:
            for result in results:
                msg: str = f"Parameters: {result[0]}, Target Value: {result[1]}"
                self.output(msg)

        return results

    def run_ga_optimization(
        self,
        optimization_setting: OptimizationSetting,
        max_workers: int = None,
        ngen_size: int = 30,
        output=True
    ) -> list:
        """Run genetic algorithm optimization"""
        if not check_optimization_setting(optimization_setting):
            return

        temp_path: Path = get_file_path("history_temp")
        with open(temp_path, mode="wb") as f:
            pickle.dump(self.history_data, f)

        evaluate_func: callable = wrap_evaluate(self, optimization_setting.target_name)
        results: list = run_ga_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            ngen_size=ngen_size,
            output=self.output
        )

        temp_path.unlink()

        if output:
            for result in results:
                msg: str = f"Parameters: {result[0]}, Target Value: {result[1]}"
                self.output(msg)

        return results

    def get_all_trades(self) -> list[TradeData]:
        """Get all trade data"""
        return list(self.trades.values())

    def get_all_orders(self) -> list[OrderData]:
        """Get all order data"""
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> list["PortfolioDailyResult"]:
        """Get all daily pnl data"""
        return list(self.daily_results.values())

    def _update_daily_close(self, bars: dict[str, BarData], dt: datetime) -> None:
        """Update daily close prices"""
        d: date = dt.date()

        close_prices: dict = {}
        for bar in bars.values():
            close_prices[bar.vt_symbol] = bar.close_price

        daily_result: Optional[PortfolioDailyResult] = self.daily_results.get(d, None)

        if daily_result:
            daily_result.update_close_prices(close_prices)
        else:
            self.daily_results[d] = PortfolioDailyResult(d, close_prices)

    def _new_bars(self, dt: datetime) -> None:
        """New bars update"""
        self.datetime = dt
        self.bars: dict[str, BarData] = self.history_data[dt]

        self._cross_limit_order()
        self.strategy.on_bars(self.bars)

        if self.strategy.inited:
            self._update_daily_close(self.bars, dt)

    def _cross_limit_order(self) -> None:
        """Cross limit orders"""
        for order in list(self.active_limit_orders.values()):
            bar: BarData = self.bars[order.vt_symbol]

            long_cross_price: float = bar.low_price
            short_cross_price: float = bar.high_price
            long_best_price: float = bar.open_price
            short_best_price: float = bar.open_price

            # Push not traded status update
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.update_order(order)

            # Check if order can be crossed
            long_cross: bool = (
                order.direction == Direction.LONG
                and order.price >= long_cross_price
                and long_cross_price > 0
            )

            short_cross: bool = (
                order.direction == Direction.SHORT
                and order.price <= short_cross_price
                and short_cross_price > 0
            )

            if not long_cross and not short_cross:
                continue

            # Push traded status update
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.update_order(order)

            if order.vt_orderid in self.active_limit_orders:
                self.active_limit_orders.pop(order.vt_orderid)

            # Push trade data
            self.trade_count += 1

            if long_cross:
                trade_price = min(order.price, long_best_price)
            else:
                trade_price = max(order.price, short_best_price)

            trade: TradeData = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            self.strategy.update_trade(trade)
            self.trades[trade.vt_tradeid] = trade

    def load_bars(
        self,
        strategy: StrategyTemplate,
        days: int,
        interval: Interval
    ) -> None:
        """Load history bar data"""
        self.days = days

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
        price: float = round_to(price, self.priceticks[vt_symbol])
        symbol, exchange = extract_vt_symbol(vt_symbol)

        self.limit_order_count += 1

        order: OrderData = OrderData(
            symbol=symbol,
            exchange=exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            datetime=self.datetime,
            gateway_name=self.gateway_name,
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return order.vt_orderid

    def cancel_order(self, strategy: StrategyTemplate, vt_orderid: str) -> None:
        """Cancel existing order"""
        if vt_orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.update_order(order)

    def write_log(self, msg: str, strategy: StrategyTemplate = None) -> None:
        """Write log message"""
        msg: str = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def sync_strategy_data(self, strategy: StrategyTemplate) -> None:
        """Sync strategy data into file"""
        pass

    def get_pricetick(self, strategy: StrategyTemplate, vt_symbol: str) -> float:
        """Get pricetick of a contract"""
        return self.priceticks[vt_symbol]

    def get_size(self, strategy: StrategyTemplate, vt_symbol: str) -> float:
        """Get size of a contract"""
        return self.sizes[vt_symbol]

    def put_strategy_event(self, strategy: StrategyTemplate) -> None:
        """Put strategy UI update event"""
        pass

    def output(self, msg: str) -> None:
        """Output backtesting engine message"""
        print(f"{datetime.now()}\t{msg}")


class ContractDailyResult:
    """Daily pnl of each contract"""

    def __init__(self, result_date: date, close_price: float) -> None:
        """"""
        self.date: date = result_date
        self.close_price: float = close_price
        self.pre_close: float = 0

        self.trades: list[TradeData] = []
        self.trade_count: int = 0

        self.start_pos: float = 0
        self.end_pos: float = 0

        self.turnover: float = 0
        self.commission: float = 0
        self.slippage: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """Add trade data"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: int,
        rate: float,
        slippage: float
    ) -> None:
        """Calculate today pnl"""
        # Use 1 for pre_close if not found to avoid ZeroDivisionError
        if pre_close:
            self.pre_close = pre_close
        else:
            self.pre_close = 1

        # Calculate holding pnl
        self.start_pos = start_pos
        self.end_pos = start_pos

        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        # Calculate trading pnl
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume

            self.end_pos += pos_change

            turnover: float = trade.volume * size * trade.price

            self.trading_pnl += pos_change * (self.close_price - trade.price) * size
            self.slippage += trade.volume * size * slippage
            self.turnover += turnover
            self.commission += turnover * rate

        # Calculate daily pnl
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage

    def update_close_price(self, close_price: float) -> None:
        """Update daily close price"""
        self.close_price = close_price


class PortfolioDailyResult:
    """Daily pnl of the portfolio"""

    def __init__(self, result_date: date, close_prices: dict[str, float]) -> None:
        """"""
        self.date: date = result_date
        self.close_prices: dict[str, float] = close_prices
        self.pre_closes: dict[str, float] = {}
        self.start_poses: dict[str, float] = {}
        self.end_poses: dict[str, float] = {}

        self.contract_results: dict[str, ContractDailyResult] = {}

        for vt_symbol, close_price in close_prices.items():
            self.contract_results[vt_symbol] = ContractDailyResult(result_date, close_price)

        self.trade_count: int = 0
        self.turnover: float = 0
        self.commission: float = 0
        self.slippage: float = 0
        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """Add trade data"""
        contract_result: ContractDailyResult = self.contract_results[trade.vt_symbol]
        contract_result.add_trade(trade)

    def calculate_pnl(
        self,
        pre_closes: dict[str, float],
        start_poses: dict[str, float],
        sizes: dict[str, float],
        rates: dict[str, float],
        slippages: dict[str, float],
    ) -> None:
        """Calculate today pnl"""
        self.pre_closes = pre_closes
        self.start_poses = start_poses

        for vt_symbol, contract_result in self.contract_results.items():
            contract_result.calculate_pnl(
                pre_closes.get(vt_symbol, 0),
                start_poses.get(vt_symbol, 0),
                sizes[vt_symbol],
                rates[vt_symbol],
                slippages[vt_symbol]
            )

            self.trade_count += contract_result.trade_count
            self.turnover += contract_result.turnover
            self.commission += contract_result.commission
            self.slippage += contract_result.slippage
            self.trading_pnl += contract_result.trading_pnl
            self.holding_pnl += contract_result.holding_pnl
            self.total_pnl += contract_result.total_pnl
            self.net_pnl += contract_result.net_pnl

            self.end_poses[vt_symbol] = contract_result.end_pos

    def update_close_prices(self, close_prices: dict[str, float]) -> None:
        """Update close prices"""
        self.close_prices.update(close_prices)

        for vt_symbol, close_price in close_prices.items():
            contract_result: Optional[ContractDailyResult] = self.contract_results.get(vt_symbol, None)
            if contract_result:
                contract_result.update_close_price(close_price)
            else:
                self.contract_results[vt_symbol] = ContractDailyResult(self.date, close_price)


@lru_cache(maxsize=999)
def load_bar_data(
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    end: datetime
) -> list[BarData]:
    """Load bar data from database"""
    symbol, exchange = extract_vt_symbol(vt_symbol)

    database: BaseDatabase = get_database()

    return database.load_bar_data(symbol, exchange, interval, start, end)


def evaluate(
    target_name: str,
    strategy_class: StrategyTemplate,
    vt_symbols: list[str],
    interval: Interval,
    start: datetime,
    rates: dict[str, float],
    slippages: dict[str, float],
    sizes: dict[str, float],
    priceticks: dict[str, float],
    capital: int,
    end: datetime,
    setting: dict
) -> tuple:
    """Wrap the entire bacaktesting process for multiprocessing task"""
    engine: BacktestingEngine = BacktestingEngine()

    engine.set_parameters(
        interval=interval,
        start=start,
        end=end,
        capital=capital,
    )

    for vt_symbol in vt_symbols:
        engine.add_contract(
            vt_symbol,
            priceticks[vt_symbol],
            sizes[vt_symbol],
            rates[vt_symbol],
            slippages[vt_symbol]
        )

    engine.add_strategy(strategy_class, setting)

    with open(get_file_path("history_temp"), mode="rb") as f:
        engine.history_data = pickle.load(f)

    engine.run_backtesting(disable_tqdm=True)
    engine.calculate_result()
    statistics: dict = engine.calculate_statistics(output=False)

    target_value: float = statistics[target_name]
    return (str(setting), target_value, statistics)


def wrap_evaluate(engine: BacktestingEngine, target_name: str) -> callable:
    """Wrap the entire bacaktesting process for multiprocessing task"""
    func: callable = partial(
        evaluate,
        target_name,
        engine.strategy_class,
        engine.vt_symbols,
        engine.interval,
        engine.start,
        engine.rates,
        engine.slippages,
        engine.sizes,
        engine.priceticks,
        engine.capital,
        engine.end
    )
    return func


def get_target_value(result: list) -> float:
    """Get target value of optimization"""
    return result[1]
