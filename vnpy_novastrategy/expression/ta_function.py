"""
技术算子
"""

import pandas as pd
import talib


def ts_dema(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的双指数移动平均"""
    return talib.DEMA(x, window)


def ts_kama(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的自适应移动平均"""
    return talib.KAMA(x, window)


def ts_mid_point(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的中位数"""
    return talib.MIDPOINT(x, window)


def ts_beta(x1: pd.Series, x2: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的两个变量回归beta"""
    return talib.BETA(x1, x2, window)


def ts_lr_angle(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的回归角度"""
    return talib.LINEARREG_ANGLE(x, window)


def ts_lr_intercept(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的回归截距"""
    return talib.LINEARREG_INTERCEPT(x, window)


def ts_lr_slope(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的回归斜率"""
    return talib.LINEARREG_SLOPE(x, window)


def ts_ema(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的指数移动平均"""
    return talib.EMA(x, window)


def ts_wma(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的加权移动平均"""
    return talib.WMA(x, window)


def ts_cmo(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的钱德动量摆动指标"""
    return talib.CMO(x, window)


def ts_mom(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的上升动向值"""
    return talib.MOM(x, window)


def ts_roc(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的变动率指标"""
    return talib.ROC(x, window)


def ts_rocr(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的变动百分率"""
    return talib.ROCR(x, window)


def ts_rocp(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的变动百分比"""
    return talib.ROCP(x, window)


def ts_rocr100(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的变动百分率（乘以100%）"""
    return talib.ROCR100(x, window)


def ts_trix(x: pd.Series, window: int) -> pd.Series:
    """时序滚动窗口的ROC三重移动平滑指数移动平均"""
    return talib.TRIX(x, window)
