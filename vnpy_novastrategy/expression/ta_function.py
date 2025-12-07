"""
技术算子
"""

import talib
import pandas as pd


ta_dema = talib.DEMA
ta_kama = talib.KAMA
ta_rsi = talib.RSI
ta_macd = talib.MACD
ta_ema = talib.EMA
ta_ma = talib.WMA
ta_mfi = talib.MFI
ta_cmo = talib.CMO
ta_mom = talib.MOM
ta_roc = talib.ROC


def ta_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate ATR"""
    return talib.ATR(high.fillna(0), low.fillna(0), close.fillna(0), window)    # type: ignore
