from typing import Sequence

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from alphainspect import _QUANTILE_


def plot_quantile_returns_bar(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str],
                              *,
                              ax=None):
    """分组收益柱状图

    Examples
    --------
    >>> plot_quantile_returns_bar(df_pl, 'GP_0000', ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])
    """
    df_pl = df_pl.group_by(by=[_QUANTILE_]).agg([pl.mean(y) for y in forward_returns]).sort(_QUANTILE_)
    df_pd = df_pl.to_pandas().set_index(_QUANTILE_)
    ax = df_pd.plot.bar(ax=ax)
    ax.set_title(f'{factor},Mean Return By Factor Quantile')
    ax.set_xlabel('')


def plot_quantile_returns_box(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str], *, ax=None):
    """分组收益

    Examples
    --------
    >>> plot_quantile_returns_box(df_pl, 'GP_0000', ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])

    """
    df_pl = df_pl.select(_QUANTILE_, *forward_returns)
    df_pd = df_pl.to_pandas().set_index(_QUANTILE_)

    df_pd = df_pd.stack().reset_index()
    df_pd.columns = ['x', 'hue', 'y']
    df_pd = df_pd.sort_values(by=['x', 'hue'])
    ax = sns.boxplot(data=df_pd, x='x', y='y', hue='hue', ax=ax)
    ax.set_title(f'{factor}, Return By Factor Quantile')
    ax.set_xlabel('')


def create_returns_sheet(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str]):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # 一定要过滤null才能用
    df_pl = df_pl.filter(pl.col(_QUANTILE_).is_not_null())
    plot_quantile_returns_bar(df_pl, factor, forward_returns, ax=axes[0])
    plot_quantile_returns_box(df_pl, factor, forward_returns, ax=axes[1])

    fig.tight_layout()
