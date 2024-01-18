from typing import Sequence

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from alphainspect.utils import cs_bucket


def calc_returns_by_quantile(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str], quantiles: int = 10,
                             *,
                             date: str = 'date') -> pl.DataFrame:
    """收益率按因子分组

    Examples
    --------
    >>> calc_returns_by_quantile(df_pl, 'GP_0000', ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])
    """

    def _func_cs(df: pl.DataFrame):
        return df.select([
            date,
            cs_bucket(pl.col(factor), quantiles),
            *forward_returns,
        ])

    return df_pl.group_by(by=date).map_groups(_func_cs)


def plot_quantile_returns_bar(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str],
                              *,
                              ax=None):
    """分组收益柱状图

    Examples
    --------
    >>> plot_quantile_returns_bar(df_pl, 'GP_0000', ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])
    """
    df_pl = df_pl.group_by(by=factor).agg([pl.mean(y) for y in forward_returns]).sort(factor)
    df_pd = df_pl.to_pandas().set_index(factor)
    ax = df_pd.plot.bar(ax=ax)
    ax.set_title(f'{factor},Mean Return By Factor Quantile')
    ax.set_xlabel('')


def plot_quantile_returns_violin(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str], *, ax=None):
    """分组收益小提琴图

    Examples
    --------
    >>> plot_quantile_returns_violin(df_pl, 'GP_0000', ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])

    Notes
    -----
    速度有点慢
    """
    df_pd = df_pl.to_pandas().set_index(factor)[forward_returns]
    # TODO 超大数据有必要截断吗
    if len(df_pl) > 5000 * 250:
        df_pl = df_pl.sample(5000 * 120)
    df_pd = df_pd.stack().reset_index()
    df_pd.columns = ['x', 'hue', 'y']
    df_pd = df_pd.sort_values(by=['x', 'hue'])
    ax = sns.violinplot(data=df_pd, x='x', y='y', hue='hue', ax=ax)
    ax.set_title(f'{factor}, Return By Factor Quantile')
    ax.set_xlabel('')


def create_returns_sheet(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str], quantiles: int = 10,
                         *,
                         date: str = 'date'):
    df_pl = calc_returns_by_quantile(df_pl, factor, forward_returns, quantiles=quantiles, date=date)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    plot_quantile_returns_bar(df_pl, factor, forward_returns, ax=axes[0])
    plot_quantile_returns_violin(df_pl, factor, forward_returns, ax=axes[1])
