"""
时序算子
"""

import pandas as pd
from scipy import stats


def ts_min(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的最小值"""
    return x.rolling(window).min()


def ts_max(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的最大值"""
    return x.rolling(window).max()


def ts_delay(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口固定时间之前的值"""
    return x.shift(window).values


def ts_delta(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口固定时间之前的变动"""
    return x - ts_delay(x, window)


def ts_argmin(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口中最小值出现的位置"""
    return x.rolling(window).apply(lambda x: x.argmin())


def ts_argmax(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口中最大值出现的位置"""
    return x.rolling(window).apply(lambda x: x.argmax())


def ts_rank(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口中当前值所处分位数"""
    return x.rolling(window).apply(lambda x: stats.percentileofscore(x, x.tail(1)) / 100)


def ts_sum(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口之和"""
    return x.rolling(window).sum()


def ts_std(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的标准差"""
    return x.rolling(window).std()


def ts_corr(x1: pd.Series, x2: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的X1值构成的时序数列与X2构成的时序数列的相关系数"""
    return x1.rolling(window).corr(x2)


def ts_mean(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的平均值"""
    return x.rolling(window).mean()


def ts_skew(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的偏度"""
    return x.rolling(window).skew()


def ts_kurtosis(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的峰度"""
    return x.rolling(window).kurt()


def ts_greater_than(x1: pd.Series, x2: pd.Series, window: int) -> pd.Series:
    """比较X1是否大于等于X2，返回对应的0或者1"""
    x1 = x1.fillna(0)
    x2 = x2.fillna(0)
    return (x1 > x2).astype(float)


def ts_less_than(x1: pd.Series, x2: pd.Series, window: int) -> pd.Series:
    """比较X1是否小于等于X2，返回对应的0或者1"""
    x1 = x1.fillna(0)
    x2 = x2.fillna(0)
    return (x1 < x2).astype(float)


def ts_compare_mean(x: pd.Series, window: int) -> pd.Series:
    """比较X是否大于自己滚动窗口的均值"""
    mean: pd.Series = x.rolling(window).mean()
    return (x > mean).astype(float)
