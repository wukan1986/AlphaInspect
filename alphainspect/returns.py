from typing import Sequence

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from alphainspect import _QUANTILE_
from alphainspect.utils import plot_heatmap


def plot_quantile_returns_bar(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str],
                              *,
                              factor_quantile: str = _QUANTILE_,
                              ax=None):
    """分组收益柱状图

    Examples
    --------
    >>> plot_quantile_returns_bar(df_pl, 'GP_0000', ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])
    """
    df_pl = df_pl.group_by(factor_quantile).agg([pl.mean(y) for y in forward_returns]).sort(factor_quantile)
    df_pd = df_pl.to_pandas().set_index(factor_quantile)
    ax = df_pd.plot.bar(ax=ax)
    ax.set_title(f'{factor},Mean Return By Factor Quantile')
    ax.set_xlabel('')
    # ax.bar_label(ax.containers[0])


def plot_quantile_returns_box(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str],
                              *, factor_quantile: str = _QUANTILE_, ax=None):
    """分组收益

    Examples
    --------
    >>> plot_quantile_returns_box(df_pl, 'GP_0000', ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])

    """
    df_pl = df_pl.select(factor_quantile, *forward_returns)
    df_pd = df_pl.to_pandas().set_index(factor_quantile)

    df_pd = df_pd.stack().reset_index()
    df_pd.columns = ['x', 'hue', 'y']
    df_pd = df_pd.sort_values(by=['x', 'hue'])
    ax = sns.boxplot(data=df_pd, x='x', y='y', hue='hue', ax=ax)
    ax.set_title(f'{factor}, Return By Factor Quantile')
    ax.set_xlabel('')


def create_returns_sheet(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str], factor_quantile: str = _QUANTILE_):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # 一定要过滤null才能用
    df_pl = df_pl.filter(pl.col(factor_quantile).is_not_null())
    plot_quantile_returns_bar(df_pl, factor, forward_returns, factor_quantile=factor_quantile, ax=axes[0])
    plot_quantile_returns_box(df_pl, factor, forward_returns, factor_quantile=factor_quantile, ax=axes[1])

    fig.tight_layout()


def create_returns2_sheet(df_pl: pl.DataFrame,
                          forward_return: str,
                          factor_quantiles: Sequence[str]):
    """独立双重排序法。例如，将两个因子划分成5*5，查看两因子组合效果

    Parameters
    ----------
    df_pl
    forward_return
    factor_quantiles

    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))

    df_pl = df_pl.filter([pl.col(q).is_not_null() for q in factor_quantiles])
    df_mean = df_pl.group_by(factor_quantiles).agg(pl.mean(forward_return)).sort(factor_quantiles).to_pandas()
    df_std = df_pl.group_by(factor_quantiles).agg(pl.std(forward_return, ddof=0)).sort(factor_quantiles).to_pandas()
    df_mean = df_mean.set_index(factor_quantiles)[forward_return].unstack()
    df_std = df_std.set_index(factor_quantiles)[forward_return].unstack()

    plot_heatmap(df_mean, title='Mean', ax=axes[0])
    plot_heatmap(df_std, title='Std', ax=axes[1])

    fig.tight_layout()
