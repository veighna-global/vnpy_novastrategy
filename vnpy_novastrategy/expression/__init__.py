import pandas as pd

from .ts_function import *
from .cs_function import *
from .ta_function import *


def calculate_by_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    """基于表达式进行计算"""
    d: dict[str, pd.Series] = {}

    for column in df.columns:
        d[column] = df[column]

    feature = eval(expression, locals=d)

    return feature
