# 换手率
from typing import Sequence

import pandas as pd
import polars as pl
from matplotlib import pyplot as plt

from alphainspect.utils import auto_corr, cs_bucket


def calc_auto_correlation(df_pl: pl.DataFrame, factor: str,
                          *,
                          periods: Sequence[int], date: str = 'date'):
    """计算排序自相关"""
    return df_pl.group_by(by=[date]).agg([auto_corr(factor, p).alias(f'AC{p:02d}') for p in periods]).sort(date)


def _list_to_set(x):
    return set() if x is None else set(x)


def _set_diff(curr: pd.Series, period: int):
    history = curr.shift(period).apply(_list_to_set)
    new_ = (curr - history)
    # 当前持仓中有多少是新股票
    return new_.apply(len) / curr.apply(len)


def calc_quantile_turnover(df_pl: pl.DataFrame,
                           factor: str,
                           quantiles: int = 10,
                           *,
                           periods: Sequence[int] = (1, 5, 10, 20),
                           date: str = 'date', asset: str = 'asset'):
    def _func_cs(df: pl.DataFrame):
        return df.select([
            date,
            asset,
            cs_bucket(pl.col(factor), quantiles).alias('factor_quantile'),
        ])

    def _func_ts(df: pd.DataFrame, periods=periods):
        for p in periods:
            df[f'P{p:02d}'] = _set_diff(df[asset], p)
        return df

    df_pl = df_pl.group_by(by=date).map_groups(_func_cs)
    df_pd: pd.DataFrame = df_pl.group_by(by=[date, 'factor_quantile']).agg(asset).sort(date).to_pandas()
    df_pd[asset] = df_pd[asset].apply(_list_to_set)
    return df_pd.groupby(by='factor_quantile').apply(_func_ts)


def plot_factor_auto_correlation(df_pl: pl.DataFrame,
                                 *,
                                 axvlines=(), date: str = 'date', ax=None):
    df_pd = df_pl.to_pandas().set_index(date)
    ax = df_pd.plot(title='Factor Auto Correlation', cmap='coolwarm', alpha=0.7, lw=1, grid=True, ax=ax)
    for v in axvlines:
        ax.axvline(x=v, c="b", ls="--", lw=1)


def plot_turnover_quantile(df_pd: pd.DataFrame, quantile: int = 0,
                           *,
                           periods: Sequence[int] = (1, 5, 10, 20), axvlines=(), date: str = 'date', ax=None):
    df_pd = df_pd[df_pd['factor_quantile'] == quantile]
    df_pd = df_pd.set_index(date)
    df_pd = df_pd[[f'P{p:02d}' for p in periods]]
    ax = df_pd.plot(title=f'Quantile {quantile} Mean Turnover', alpha=0.7, lw=1, grid=True, ax=ax)
    for v in axvlines:
        ax.axvline(x=v, c="b", ls="--", lw=1)


def create_turnover_sheet(df, factor, quantiles: int = 10,
                          *,
                          periods: Sequence[int] = (1, 5, 10, 20), date: str = 'date', axvlines=()):
    df1 = calc_auto_correlation(df, factor, periods=periods, date=date)
    df2 = calc_quantile_turnover(df, factor, quantiles=quantiles, periods=periods)

    fix, axes = plt.subplots(2, 1, figsize=(12, 9))
    plot_factor_auto_correlation(df1, axvlines=axvlines, date=date, ax=axes[0])
    groups = (0, quantiles - 1)
    for i, q in enumerate(groups):
        ax = plt.subplot(223 + i)
        plot_turnover_quantile(df2, quantile=q, periods=periods, axvlines=axvlines, date=date, ax=ax)
