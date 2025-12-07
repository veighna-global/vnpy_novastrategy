"""
时序算子
"""

import pandas as pd
from scipy import stats


def ts_min(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的最小值"""
    result: pd.Series = x.groupby(level=1).rolling(window).min()
    return result.reset_index(level=0, drop=True)


def ts_max(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的最大值"""
    result: pd.Series = x.groupby(level=1).rolling(window).max()
    return result.reset_index(level=0, drop=True)


def ts_delay(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口固定时间之前的值"""
    return x.shift(window)


def ts_delta(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口固定时间之前的变动"""
    return x - ts_delay(x, window)


def ts_rank(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口中当前值所处分位数"""
    result: pd.Series = x.groupby(level=1).rolling(window).apply(lambda x: stats.percentileofscore(x, x.tail(1)) / 100)
    return result.reset_index(level=0, drop=True)


def ts_sum(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口之和"""
    result: pd.Series = x.groupby(level=1).rolling(window).sum()
    return result.reset_index(level=0, drop=True)


def ts_std(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的标准差"""
    result: pd.Series = x.groupby(level=1).rolling(window).std()
    return result.reset_index(level=0, drop=True)


def ts_corr(x1: pd.Series, x2: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的X1值构成的时序数列与X2构成的时序数列的相关系数"""
    return x1.rolling(window).corr(x2)


def ts_mean(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的平均值"""
    result: pd.Series = x.groupby(level=1).rolling(window).mean()
    return result.reset_index(level=0, drop=True)


def ts_skew(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的偏度"""
    result: pd.Series = x.groupby(level=1).rolling(window).skew()
    return result.reset_index(level=0, drop=True)


def ts_kurtosis(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的峰度"""
    result: pd.Series = x.groupby(level=1).rolling(window).kurt()
    return result.reset_index(level=0, drop=True)


def ts_greater_than(x1: pd.Series, x2: pd.Series | float) -> pd.Series:
    """比较X1是否大于等于X2，返回对应的0或者1"""
    x1 = x1.fillna(0)

    if isinstance(x2, pd.Series):
        x2 = x2.fillna(0)

    return (x1 > x2).astype(float)


def ts_less_than(x1: pd.Series, x2: pd.Series | float) -> pd.Series:
    """比较X1是否小于等于X2，返回对应的0或者1"""
    x1 = x1.fillna(0)

    if isinstance(x2, pd.Series):
        x2 = x2.fillna(0)

    return (x1 < x2).astype(float)
