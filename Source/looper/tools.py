import numpy as np
import pandas as pd
from typing import Iterable, Optional, List


def sets_match(a: Iterable, b: Iterable):
    return a is None or set(a) == set(b)


def df_set(self: pd.DataFrame, new: pd.DataFrame):
    new_index = set(new.index).difference(self.index)
    new_columns = set(new.columns).difference(self.columns)
    for column in new_columns: self[column] = np.nan
    if len(self.columns) > 0:
        for key in new_index: self.loc[key] = np.nan
    if len(new.columns) == 0 or len(new.index) == 0: return
    self.loc[list(new.index), list(new.columns)] = new


def df_get(self: pd.DataFrame, cols: Optional[List[str]], index: Optional[List[str]]):
    if cols is None: cols = self.columns
    if index is None: index = self.index
    if len(cols) == 0 or len(index) == 0: return pd.DataFrame()
    return self.loc[list(index), list(cols)]


pd.DataFrame.df_set = df_set
pd.DataFrame.df_get = df_get
