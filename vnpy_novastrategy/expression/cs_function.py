import pandas as pd


def rank(x: pd.Series) -> pd.Series:
    """对因子数据按照时间戳进行截面排序"""
    return x.groupby(level=0).rank(
        method="average",
        ascending=True,
        na_option="bottom",
        pct=True
    )


def normalize(x: pd.Series) -> pd.Series:
    """对因子数据按照时间戳进行归一化"""
    x_mean = x.groupby(level=0).mean()
    return x - x_mean


def cs_sum(x: pd.Series) -> pd.Series:
    """对因子数据按照时间戳进行求和"""
    x_sum = x.groupby(level=0).sum()
    return x_sum.reindex(x.index, level=0)


def cs_count(x: pd.Series) -> pd.Series:
    """对因子数据按照时间戳进行计数"""
    x_count = x.groupby(level=0).count()
    return x_count.reindex(x.index, level=0)
