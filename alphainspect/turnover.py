# 换手率
from typing import Sequence

import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from polars import Expr

from alphainspect import _QUANTILE_, _DATE_, _ASSET_


def auto_corr(a: str, period: int) -> Expr:
    """自相关"""
    return pl.corr(pl.col(a), pl.col(a).shift(period), method='spearman', ddof=0, propagate_nans=False)


def calc_auto_correlation(df_pl: pl.DataFrame,
                          factor: str,
                          *,
                          periods: Sequence[int]):
    """计算排序自相关"""
    return df_pl.group_by(_DATE_).agg([auto_corr(factor, p).alias(f'AC{p:02d}') for p in periods]).sort(_DATE_)


def _list_to_set(x):
    return set() if x is None else set(x)


def _set_diff(curr: pd.Series, period: int):
    history = curr.shift(period).apply(_list_to_set)
    new_ = (curr - history)
    # 当前持仓中有多少是新股票
    return new_.apply(len) / curr.apply(len)


def calc_quantile_turnover(df_pl: pl.DataFrame,
                           *,
                           factor_quantile: str = _QUANTILE_,
                           periods: Sequence[int] = (1, 5, 10, 20)) -> pd.DataFrame:
    def _func_ts(df: pd.DataFrame, periods=periods):
        for p in periods:
            df[f'P{p:02d}'] = _set_diff(df[_ASSET_], p)
        return df

    df_pd: pd.DataFrame = df_pl.group_by(_DATE_, factor_quantile).agg(_ASSET_).sort(_DATE_).to_pandas()
    df_pd[_ASSET_] = df_pd[_ASSET_].apply(_list_to_set)
    return df_pd.groupby(by=factor_quantile).apply(_func_ts)


def plot_factor_auto_correlation(df_pl: pl.DataFrame,
                                 *,
                                 axvlines=(), ax=None):
    df_pd = df_pl.to_pandas().set_index(_DATE_)
    ax = df_pd.plot(title='Factor Auto Correlation', cmap='coolwarm', alpha=0.7, lw=1, grid=True, ax=ax)
    ax.set_xlabel('')
    for v in axvlines:
        ax.axvline(x=v, c="b", ls="--", lw=1)


def plot_turnover_quantile(df_pd: pd.DataFrame, quantile: int,
                           *,
                           factor_quantile: str = _QUANTILE_,
                           periods: Sequence[int] = (1, 5, 10, 20), axvlines=(), ax=None):
    df_pd = df_pd[df_pd[factor_quantile] == quantile]
    df_pd = df_pd.set_index(_DATE_)
    df_pd = df_pd[[f'P{p:02d}' for p in periods]]
    ax = df_pd.plot(title=f'Quantile {quantile} Mean Turnover', alpha=0.7, lw=1, grid=True, ax=ax)
    ax.set_xlabel('')
    for v in axvlines:
        ax.axvline(x=v, c="b", ls="--", lw=1)


def create_turnover_sheet(df, factor,
                          *,
                          factor_quantile: str = _QUANTILE_,
                          periods: Sequence[int] = (1, 5, 10, 20), axvlines=()):
    df1 = calc_auto_correlation(df, factor, periods=periods)
    df2 = calc_quantile_turnover(df, periods=periods, factor_quantile=factor_quantile)
    q_min, q_max = df2[factor_quantile].min(), df2[factor_quantile].max()

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    plot_factor_auto_correlation(df1, axvlines=axvlines, ax=axes[0])

    for i, q in enumerate((q_min, q_max)):
        ax = plt.subplot(223 + i)
        plot_turnover_quantile(df2, quantile=q, periods=periods, factor_quantile=factor_quantile, axvlines=axvlines, ax=ax)

    fig.tight_layout()
