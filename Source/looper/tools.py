import numpy as np
import pandas as pd
from typing import Iterable, Optional, List


def sets_match(a: Iterable, b: Iterable):
    return a is None or set(a) == set(b)


def sub_set(self: pd.DataFrame, new: pd.DataFrame):
    new_index = set(new.index).difference(self.index)
    new_columns = set(new.columns).difference(self.columns)
    for column in new_columns: self[column] = np.nan
    if len(self.columns) > 0:
        for key in new_index: self.loc[key] = np.nan
    if len(new.columns) == 0 or len(new.index) == 0: return
    self.loc[list(new.index), list(new.columns)] = new


def sub_get(self: pd.DataFrame, cols: Optional[List[str]], index: Optional[List[str]]):
    if cols is None: cols = self.columns
    if index is None: index = self.index
    if len(cols) == 0 or len(index) == 0: return pd.DataFrame()
    return self.loc[list(index), list(cols)]


def series_and(s0: pd.Series, s1: pd.Series):
    index = set(s0.index).intersection(s1.index)
    if len(index) == 0: return pd.Series()
    return s0[index] & s1[index]


def series_or(s0: pd.Series, s1: pd.Series):
    index = set(s0.index).union(s1.index)
    if len(index) == 0: return pd.Series()
    s = pd.Series(False, index = index)
    if len(s0.index) > 0: s[s0.index] = s[s0.index] | s0
    if len(s1.index) > 0: s[s1.index] = s[s1.index] | s1
    return s


pd.DataFrame.sub_set = sub_set
pd.DataFrame.sub_get = sub_get
