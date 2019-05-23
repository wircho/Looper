import os
import numpy as np
import pandas as pd
from tools import sets_match
from typing import List, Optional, Union


class DataSource:
    def unsafe_get(self, names: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> pd.DataFrame:
        raise NotImplementedError("get is not implemented in {}".format(self.__class__.__name__))

    def set(self, dataframe: pd.DataFrame):
        raise NotImplementedError("set is not implemented in {}".format(self.__class__.__name__))

    def get(self, names: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> pd.DataFrame:
        dataframe = self.unsafe_get(names = names, labels = labels)
        assert sets_match(names, dataframe.index), "Non-matching names in {}.get".format(self.__class__.__name__)
        assert sets_match(labels, dataframe.columns), "Non-matching labels in {}.get".format(self.__class__.__name__)
        return dataframe

    def set_value(self, names: Union[str, List[str]], labels: Union[str, List[str]], value):
        if isinstance(names, str): return self.set_value([names], labels, value)
        if isinstance(labels, str): return self.set_value(names, [labels], value)
        self.set(pd.DataFrame(value, index = names, columns = labels))


class LabelDataSource(DataSource):
    def set_true(self, names: Union[str, List[str]], labels: Union[str, List[str]]):
        return self.set_value(names, labels, 1)

    def set_false(self, names: Union[str, List[str]], labels: Union[str, List[str]]):
        return self.set_value(names, labels, 0)

    def clear(self, names: Union[str, List[str]], labels: Union[str, List[str]]):
        return self.set_value(names, labels, np.nan)


class CachedDataSource(DataSource):
    def __init__(self, dtype):
        self.dtype = dtype
        self.dataframe = self.pull()

    def unsafe_pull(self, names: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> pd.DataFrame:
        raise NotImplementedError("pull is not implemented in {}".format(self.__class__.__name__))

    def unsafe_push(self, dataframe: pd.DataFrame):
        raise NotImplementedError("push is not implemented in {}".format(self.__class__.__name__))

    def pull(self, names: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> pd.DataFrame:
        dataframe = self.unsafe_pull(names = names, labels = labels)
        if len(dataframe.columns) > 0: dataframe = dataframe.astype(self.dtype)
        assert sets_match(names, dataframe.index), "Non-matching names in {}.get".format(self.__class__.__name__)
        assert sets_match(labels, dataframe.columns), "Non-matching labels in {}.get".format(self.__class__.__name__)
        return dataframe

    def push(self, dataframe: pd.DataFrame):
        return self.unsafe_push(dataframe.astype(self.dtype))

    def refresh(self, names: Optional[List[str]] = None, labels: Optional[List[str]] = None):
        dataframe = self.pull(names = names, labels = labels)
        self.dataframe.df_set(dataframe)
        self.dataframe = self.dataframe.astype(self.dtype)

    def unsafe_get(self, names: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> pd.DataFrame:
        return self.dataframe.df_get(labels, names)

    def set(self, dataframe: pd.DataFrame):
        dataframe = dataframe.astype(self.dtype)
        dataframe = self.push(dataframe)
        assert type(dataframe) is pd.DataFrame, "push must return dataframe in {}".format(self.__class__.__name__)
        self.dataframe = dataframe


class FileDataSource(CachedDataSource):
    def __init__(self, path: str, dtype = "Int64"):
        self.path = path
        dirname = os.path.dirname(path)
        if not os.path.isdir(dirname): os.makedirs(dirname)
        if not os.path.isfile(path):
            dataframe = pd.DataFrame(dtype = dtype)
            dataframe.to_pickle(path)
        super().__init__(dtype)

    def unsafe_pull(self, names: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> pd.DataFrame:
        dataframe = pd.read_pickle(self.path)
        return dataframe.df_get(names, labels)

    def unsafe_push(self, dataframe: pd.DataFrame):
        new_dataframe = self.dataframe.copy()
        new_dataframe.df_set(dataframe)
        new_dataframe = new_dataframe.astype(self.dtype)
        new_dataframe.to_pickle(self.path)
        return new_dataframe


class FileLabelDataSource(FileDataSource, LabelDataSource):
    def __init__(self, path: str):
        super().__init__(path, dtype = "Int64")