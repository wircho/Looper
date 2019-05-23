from unittest import TestCase
import os
import pandas as pd
from data_source import FileLabelDataSource


class TestFileDataSource(TestCase):
    def test_get_set(self):
        path = "../../../Data/test.pkl"
        if os.path.isfile(path): os.remove(path)
        source = FileLabelDataSource(path)
        df0 = pd.DataFrame([[1, 2], [3, 4]], columns = ["a", "b"], index = ["x", "y"]).astype(source.dtype)
        df1 = pd.DataFrame([[5, 6], [7, 8]], columns = ["b", "c"], index = ["y", "z"]).astype(source.dtype)
        source.set(df0)
        got = source.get(labels = df0.columns, names = df0.index)
        if not got.equals(df0): self.fail("Did not set df0 properly:\n{}\n(type: {})\n{}\n(type: {})".format(got, got.dtypes, df0, df0.dtypes))
        source.set(df1)
        got = source.get(labels = df1.columns, names = df1.index)
        if not got.equals(df1): self.fail("Did not set df1 properly:\n{}\n(type: {})\n{}\n(type: {})".format(got, got.dtypes, df1, df1.dtypes))
        source1 = FileLabelDataSource(source.path)
        got = source1.get(labels = df1.columns, names = df1.index)
        if not got.equals(df1): self.fail("Did not set df1 properly:\n{}\n(type: {})\n{}\n(type: {})".format(got, got.dtypes, df1, df1.dtypes))

