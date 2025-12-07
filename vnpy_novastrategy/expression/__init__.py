from collections.abc import Callable

import pandas as pd

from .ts_function import (
    ts_min, ts_max, ts_delay, ts_delta, ts_rank, ts_sum, ts_std,
    ts_corr, ts_mean, ts_skew, ts_kurtosis, ts_greater_than, ts_less_than
)
from .cs_function import (
    rank, normalize, cs_sum, cs_count, cs_mean, cs_std, cs_zscore
)
from .ta_function import (
    ta_dema, ta_kama, ta_rsi, ta_macd, ta_ema, ta_ma, ta_mfi,
    ta_cmo, ta_mom, ta_roc, ta_atr
)


user_functions: dict[str, Callable] = {}


def calculate_by_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    """Calculate feature/label by expression"""
    global user_functions
    locals().update(user_functions)

    d: dict[str, pd.Series] = {}

    for column in df.columns:
        d[column] = df[column]

    feature = eval(expression, None, d)

    return feature


def register_function(function: Callable) -> None:
    """Register user defined function"""
    global user_functions
    user_functions[function.__name__] = function


__all__ = [
    # Time-series functions
    "ts_min", "ts_max", "ts_delay", "ts_delta", "ts_rank", "ts_sum", "ts_std",
    "ts_corr", "ts_mean", "ts_skew", "ts_kurtosis", "ts_greater_than", "ts_less_than",

    # Cross-sectional functions
    "rank", "normalize", "cs_sum", "cs_count", "cs_mean", "cs_std", "cs_zscore",

    # Technical analysis functions
    "ta_dema", "ta_kama", "ta_rsi", "ta_macd", "ta_ema", "ta_ma", "ta_mfi",
    "ta_cmo", "ta_mom", "ta_roc", "ta_atr",

    # Expression functions
    "calculate_by_expression", "register_function",
]
