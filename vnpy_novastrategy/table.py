from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd

from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval

from .expression import calculate_by_expression


class DataTable:
    """Time-series data container for crypto strategy"""

    interval_freq_map = {
        Interval.MINUTE: "min",
        Interval.HOUR: "h",
        Interval.DAILY: "d"
    }

    def __init__(
        self,
        vt_symbols: list[str],
        size: int = 100,
        window: int = 1,
        interval: Interval = Interval.MINUTE,
        extra_fields: list[str] = None
    ) -> None:
        """"""
        self.vt_symbols: list[str] = vt_symbols
        self.size: int = size
        self.window: int = window
        self.interval: Interval = interval

        if not extra_fields:
            extra_fields = []
        self.extra_fields: list[str] = extra_fields

        self.df: pd.DataFrame = None
        self.ix: int = 0
        self.inited: bool = False
        self.periods: int = size * 100

        self.feature_expressions: dict[str, str] = {}

        # initialize DataAggregator
        if interval == Interval.MINUTE:
            agg_window: int = window
        elif interval == Interval.HOUR:
            agg_window: int = window * 60
        elif interval == Interval.DAILY:
            agg_window: int = 240

        if agg_window > 1:
            self.aggregators: dict[str, DataAggregator] = {}
            for vt_symbol in vt_symbols:
                self.aggregators[vt_symbol] = DataAggregator(agg_window)

            self.update_bars = self._update_minute_bars
        else:
            self.update_bars = self._update_window_bars

    def update_bars(self, bars: dict[str, BarData]) -> bool:
        """Update bars data into table"""
        pass

    def add_feature(self, name: str, expression: str) -> None:
        """Add feature expression for querying df"""
        self.feature_expressions[name] = expression

    def _update_minute_bars(self, bars: dict[str, BarData]) -> bool:
        """Update minute bars to aggregate window bars"""
        window_bars: dict = {}

        for vt_symbol, bar in bars.items():
            bg: DataAggregator = self.aggregators[vt_symbol]
            window_bar: BarData = bg.update_bar(bar)
            if window_bar:
                window_bars[vt_symbol] = window_bar

        if window_bars:
            self._update_window_bars(bars)
            return self.inited
        else:
            return False

    def _update_window_bars(self, bars: dict[str, BarData]) -> bool:
        """Update window bars directly into table"""
        # Check DF state
        if self.df is None:
            self._init_df(bars)
        elif self.ix == self.periods:
            self._reset_df(bars)

        # Update data into DF
        df: pd.DataFrame = self.df

        for bar in bars.values():
            data: list = [
                bar.open_price,
                bar.high_price,
                bar.low_price,
                bar.close_price,
                bar.volume,
                bar.turnover,
                bar.open_interest
            ]

            for field in self.extra_fields:
                value: object = bar.extra.get(field, None)
                data.append(value)

            df.loc[(bar.datetime, bar.vt_symbol)] = data

        # Update latest index
        self.ix += 1

        # Check if inited
        if not self.inited and self.ix > self.size:
            self.inited = True

        return self.inited

    def _init_df(self, bars: dict[str, BarData]) -> None:
        """Initialize dataFrame"""
        bar: BarData = list(bars.values())[0]

        dt_index: pd.DatetimeIndex = pd.date_range(
            start=bar.datetime,
            periods=self.periods,
            freq=self.interval_freq_map[self.interval]
        )

        multi_index: pd.MultiIndex = pd.MultiIndex.from_product(
            [dt_index, self.vt_symbols],
            names=["datetime", "vt_symbol"]
        )

        columns: list[str] = [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "turnover",
            "open_interest"
        ]
        columns += self.extra_fields

        self.df = pd.DataFrame(
            np.zeros((len(multi_index), len(columns))),
            index=multi_index,
            columns=columns
        )

        self.ix = 0

    def _reset_df(self, bars: dict[str, BarData]) -> None:
        """Reset dataFrame"""
        old_df: pd.DataFrame = self.df
        dt_index: pd.DatetimeIndex = old_df.index.levels[0]

        start: pd.Timestamp = dt_index[-self.size]

        dt_index: pd.DatetimeIndex = pd.date_range(
            start=start,
            periods=self.periods,
            freq=self.interval_freq_map[self.interval]
        )

        multi_index: pd.MultiIndex = pd.MultiIndex.from_product(
            [dt_index, self.vt_symbols],
            names=["datetime", "vt_symbol"]
        )

        columns: list[str] = [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "turnover",
            "open_interest"
        ]
        columns += self.extra_fields

        self.df = pd.DataFrame(
            np.zeros((len(multi_index), len(columns))),
            index=multi_index,
            columns=columns
        )

        fill_ix: int = len(self.vt_symbols) * self.size
        self.df.iloc[:fill_ix] = old_df.iloc[-fill_ix:]

        self.ix = self.size

    def get_df(self) -> Optional[pd.DataFrame]:
        """Get current dataframe"""
        if self.df is None:
            return None

        symbol_count: int = len(self.vt_symbols)
        end_ix: int = self.ix * symbol_count
        start_ix: int = max(self.ix - self.size, 0) * symbol_count

        df: pd.DataFrame = self.df.iloc[start_ix: end_ix].copy()

        for name, expression in self.feature_expressions.items():
            df[name] = calculate_by_expression(df, expression)

        return df


class DataAggregator:
    """Bar data aggregator for crypto strategy"""

    def __init__(self, agg_window: int) -> None:
        """Constructor"""
        self.agg_window: int = agg_window
        self.window_bar: BarData = None

    def update_bar(self, bar: BarData) -> None:
        """Update 1 minute bar into aggregator"""
        # If not inited, create window bar object
        if not self.window_bar:
            dt: datetime = bar.datetime.replace(second=0, microsecond=0)
            self.window_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price
            )
            self.window_bar.extra = {}
        # Otherwise, update high/low price into window bar
        else:
            self.window_bar.high_price = max(
                self.window_bar.high_price,
                bar.high_price
            )
            self.window_bar.low_price = min(
                self.window_bar.low_price,
                bar.low_price
            )

        # Update close price/volume/turnover into window bar
        self.window_bar.close_price = bar.close_price
        self.window_bar.volume += bar.volume
        self.window_bar.turnover += bar.turnover
        self.window_bar.open_interest = bar.open_interest

        # Sum up extra fields
        if bar.extra:
            for k, v in bar.extra.items():
                window_v: float = self.window_bar.extra.get(k, 0) + v
                self.window_bar.extra[k] = window_v

        # Check if window bar completed
        window_bar: BarData = None

        minute_passed: int = bar.datetime.hour * 60 + bar.datetime.minute
        if not (minute_passed + 1) % self.agg_window:
            window_bar = self.window_bar
            self.window_bar = None

        return window_bar
