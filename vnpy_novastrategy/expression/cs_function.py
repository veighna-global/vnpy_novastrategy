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
    x_mean: pd.Series = x.groupby(level=0).mean()
    return x - x_mean


def cs_sum(x: pd.Series) -> pd.Series:
    """对因子数据按照时间戳进行求和"""
    x_sum: pd.Series = x.groupby(level=0).sum()
    return x_sum.reindex(x.index, level=0)


def cs_count(x: pd.Series) -> pd.Series:
    """对因子数据按照时间戳进行计数"""
    x_count: pd.Series = x.groupby(level=0).count()
    return x_count.reindex(x.index, level=0)


def cs_mean(x: pd.Series) -> pd.Series:
    """对因子数据按照时间戳求均值"""
    x_mean: pd.Series = x.groupby(level=0).mean()
    return x_mean.reindex(x.index, level=0)


def cs_std(x: pd.Series) -> pd.Series:
    """对因子数据按照时间戳求标准差"""
    x_std: pd.Series = x.groupby(level=0).std()
    return x_std.reindex(x.index, level=0)


def cs_zscore(x: pd.Series) -> pd.Series:
    """对因子数据按照时间戳求Z-Score"""
    x_group: pd.core.groupby.SeriesGroupBy = x.groupby(level=0)
    x_mean: pd.Series = x_group.mean()
    x_std: pd.Series = x_group.std()
    x_zcore: pd.Series = (x - x_mean) / x_std
    return x_zcore
