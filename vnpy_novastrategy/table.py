from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd

from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval


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
        interval: Interval = Interval.MINUTE,
        extra_fields: list[str] = None
    ) -> None:
        """"""
        self.vt_symbols: list[str] = vt_symbols
        self.size: int = size
        self.interval: Interval = interval

        if not extra_fields:
            extra_fields = []
        self.extra_fields: list[str] = extra_fields

        self.df: pd.DataFrame = None
        self.ix: int = 0
        self.inited: bool = False
        self.periods: int = size * 100
        self.dt: datetime = None

    def update_bars(self, bars: dict[str, BarData]) -> None:
        """Update bars data"""
        # Check DF state
        if self.df is None:
            self.init_df(bars)
        elif self.ix == self.periods:
            self.reset_df(bars)

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

        # Update latest dastetime
        self.dt = bar.datetime

        # Update latest index
        self.ix += 1

        # Check if inited
        if not self.inited and self.ix > self.size:
            self.inited = True

    def init_df(self, bars: dict[str, BarData]) -> None:
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

    def reset_df(self, bars: dict[str, BarData]) -> None:
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
        return self.df.iloc[start_ix: end_ix]

    def get_dt(self) -> datetime:
        """Get the datetime of latest bar"""
        return self.dt
