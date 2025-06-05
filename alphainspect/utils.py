from typing import Sequence, List

import numpy as np
import pandas as pd
import polars as pl
from polars import selectors as cs
from polars_ta.wq import cs_qcut, cs_top_bottom

from alphainspect import _QUANTILE_, _DATE_


def with_factor_quantile(df: pl.DataFrame, factor: str, quantiles: int = 9, by: List[str] = None, factor_quantile: str = _QUANTILE_) -> pl.DataFrame:
    """添加因子分位数信息

    Parameters
    ----------
    df
    factor
        因子名
    quantiles
        分层数
    by
        分组
    factor_quantile
        分组名

    """
    if by is None:
        by = [_DATE_]
    return df.with_columns(cs_qcut(pl.col(factor).fill_nan(None), quantiles).over(*by).alias(factor_quantile))


def with_factor_top_k(df: pl.DataFrame, factor: str, top_k: int = 9, by: List[str] = None, factor_quantile: str = _QUANTILE_) -> pl.DataFrame:
    """前K后K的分层方法。一般用于截面股票数不多无法分位数分层的情况。

    Parameters
    ----------
    df
    factor
    top_k
    by
    factor_quantile

    Returns
    -------
    pl.DataFrame
        输出范围为0、1、2，分别为做空、对冲、做多。输出数量并不等于top_k，

        - 遇到重复时会出现数量大于top_k，
        - 遇到top_k>数量/2时，即在做多组又在做空组会划分到对冲组

    """
    if by is None:
        by = [_DATE_]

    df = df.with_columns(cs_top_bottom(pl.col(factor).fill_nan(None), top_k).over(*by).alias(factor_quantile))
    df = df.with_columns(pl.col(factor_quantile) + 1)
    return df


def select_by_suffix(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """选择指定后缀的所有因子"""
    return df.select(cs.ends_with(name).name.map(lambda x: x[:-len(name)]))


def select_by_prefix(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """选择指定前缀的所有因子"""
    return df.select(cs.starts_with(name).name.map(lambda x: x[len(name):]))


# =================================
# 没分好类的函数先放这，等以后再移动
def symmetric_orthogonal(matrix):
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # 按照特征值的大小排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 正交化矩阵
    orthogonal_matrix = np.linalg.qr(sorted_eigenvectors)[0]

    return orthogonal_matrix


def row_unstack(df: pl.DataFrame, index: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
    """一行值堆叠成一个矩阵"""
    return pd.DataFrame(df.to_numpy().reshape(len(index), len(columns)),
                        index=index, columns=columns)


def index_split_unstack(s: pd.Series, split_by: str = '__'):
    s.index = pd.MultiIndex.from_tuples(map(lambda x: tuple(x.split(split_by)), s.index))
    return s.unstack()
