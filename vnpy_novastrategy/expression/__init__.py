from typing import Callable

import pandas as pd

from .ts_function import *
from .cs_function import *
from .ta_function import *


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
