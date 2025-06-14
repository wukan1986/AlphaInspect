"""
换手率
1. 因子的自相关可以判断换手率。因子的自相关越高，换手率越低。
2. 票池判断。今集合与前集合的差集，就是换手率。

"""
from typing import Sequence

import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from polars import Expr

from alphainspect import _QUANTILE_, _DATE_, _ASSET_


def auto_corr(col: str, period: int) -> Expr:
    """自相关

    Parameters
    ----------
    col
        列名
    period
        自相关周期

    """
    return pl.corr(pl.col(col), pl.col(col).shift(period), method='spearman', ddof=0, propagate_nans=False)


def calc_auto_correlation(df: pl.DataFrame,
                          factor: str,
                          *,
                          periods: Sequence[int]):
    """计算排序自相关

    Parameters
    ----------
    df
    factor
        因子
    periods
        多个周期

    """
    return df.group_by(_DATE_).agg([auto_corr(factor, p).alias(f'AC{p:02d}') for p in periods]).sort(_DATE_)


def _list_to_set(x):
    """列表转集合"""
    return set() if x is None else set(x)


def _set_diff(curr: pd.Series, period: int):
    """集合差异。当前持仓中有多少是新股票。即换手率"""
    history = curr.shift(period).apply(_list_to_set)
    return (curr - history).apply(len) / curr.apply(len)


def calc_quantile_turnover(df: pl.DataFrame,
                           *,
                           factor_quantile: str = _QUANTILE_,
                           periods: Sequence[int] = (1, 5, 10, 20)) -> pd.DataFrame:
    """计算不同分位数,不同周期的换手率

    Parameters
    ----------
    df
    factor_quantile
        因子分位数分组
    periods
        不同周期

    """

    def _func_ts(df: pd.DataFrame, periods=periods):
        for p in periods:
            df[f'P{p:02d}'] = _set_diff(df[_ASSET_], p)
        return df

    # 集合操作在pandas上比较方便，polars中不支持
    df_pd: pd.DataFrame = df.group_by(_DATE_, factor_quantile).agg(_ASSET_).sort(_DATE_).to_pandas()
    df_pd[_ASSET_] = df_pd[_ASSET_].apply(_list_to_set)
    return df_pd.groupby(by=factor_quantile)[df_pd.columns].apply(_func_ts)


def plot_factor_auto_correlation(df: pl.DataFrame,
                                 *,
                                 axvlines=(),
                                 ax=None):
    """绘制自相关图"""
    df_pd = df.to_pandas().set_index(_DATE_)
    ax = df_pd.plot(title='Factor Auto Correlation', cmap='coolwarm', alpha=0.7, lw=1, grid=True, ax=ax)
    ax.set_xlabel('')
    for v in axvlines:
        ax.axvline(x=v, c="b", ls="--", lw=1)


def plot_turnover_quantile(df: pd.DataFrame, quantile: int,
                           *,
                           factor_quantile: str = _QUANTILE_,
                           periods: Sequence[int] = (1, 5, 10, 20),
                           axvlines=(),
                           ax=None):
    """绘制不同分位数,不同周期的换手率图

    Parameters
    ----------
    df
    quantile
        只绘制这个分位数
    factor_quantile
    periods
        不同周期
    axvlines
    ax

    """
    df = df[df[factor_quantile] == quantile]
    df = df.set_index(_DATE_)
    df = df[[f'P{p:02d}' for p in periods]]
    ax = df.plot(title=f'Quantile {quantile} Mean Turnover', alpha=0.7, lw=1, grid=True, ax=ax)
    ax.set_xlabel('')
    for v in axvlines:
        ax.axvline(x=v, c="b", ls="--", lw=1)


def create_turnover_sheet(df: pl.DataFrame, factor: str,
                          *,
                          factor_quantile: str = _QUANTILE_,
                          periods: Sequence[int] = (1, 5, 10, 20), axvlines=()):
    """绘制换手率图

    Parameters
    ----------
    df
    factor
    factor_quantile
    periods
    axvlines

    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # 换手率
    df2 = calc_quantile_turnover(df, periods=periods, factor_quantile=factor_quantile)
    q_min, q_max = df2[factor_quantile].min(), df2[factor_quantile].max()

    # 自相关
    df1 = calc_auto_correlation(df, factor, periods=periods)
    plot_factor_auto_correlation(df1, axvlines=axvlines, ax=axes[0])

    for i, q in enumerate((q_min, q_max)):
        ax = plt.subplot(223 + i)
        plot_turnover_quantile(df2, quantile=q, periods=periods, factor_quantile=factor_quantile, axvlines=axvlines, ax=ax)

    fig.tight_layout()
