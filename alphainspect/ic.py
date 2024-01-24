from typing import Sequence

import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels import api as sm

from alphainspect import _DATE_
from alphainspect.utils import rank_ic


def calc_ic(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str]) -> pl.DataFrame:
    """计算一个因子与多个标签的IC

    Parameters
    ----------
    df_pl: pl.DataFrame
    factor:str
        因子
    forward_returns:str
        标签列表

    Examples
    --------
    # doctest: +SKIP
    >>> calc_ic(df_pl, 'SMA_020', ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])

    """
    return df_pl.group_by(by=[_DATE_]).agg(
        # 这里没有换名，名字将与forward_returns对应
        [rank_ic(x, factor) for x in forward_returns]
    ).sort(_DATE_)


def plot_ic_ts(df_pl: pl.DataFrame, col: str,
               *,
               axvlines=(), ax=None) -> None:
    """IC时序图

    Examples
    --------
    >>> plot_ic_ts(df_pd, 'RETURN_OO_1')
    """
    df_pl = df_pl.select([_DATE_, col])

    df_pl = df_pl.select([
        _DATE_,
        pl.col(col).alias('ic'),
        pl.col(col).rolling_mean(20).alias('sma_20'),
        pl.col(col).fill_nan(0).cum_sum().alias('cum_sum'),
    ])

    df_pd = df_pl.to_pandas().dropna()
    s: pd.Series = df_pd['ic']

    ic = s.mean()
    ir = s.mean() / s.std()
    rate = (s.abs() > 0.02).value_counts(normalize=True).loc[True]

    ax1 = df_pd.plot.line(x=_DATE_, y=['ic', 'sma_20'], alpha=0.5, lw=1,
                          title=f"{col},IC={ic:0.4f},>0.02={rate:0.2f},IR={ir:0.4f}",
                          ax=ax)
    ax2 = df_pd.plot.line(x=_DATE_, y=['cum_sum'], alpha=0.9, lw=1,
                          secondary_y='cum_sum', c='r',
                          ax=ax1)
    ax1.axhline(y=ic, c="r", ls="--", lw=1)
    ax.set_xlabel('')
    for v in axvlines:
        ax1.axvline(x=v, c="b", ls="--", lw=1)


def plot_ic_hist(df_pl: pl.DataFrame, col: str,
                 *,
                 ax=None) -> None:
    """IC直方图

    Examples
    --------
    >>> plot_ic_hist(df_pl, 'RETURN_OO_1')
    """
    a = df_pl[col].to_pandas().dropna()

    mean = a.mean()
    std = a.std()
    skew = a.skew()
    kurt = a.kurt()

    ax = sns.histplot(a,
                      bins=50, kde=True,
                      stat="density", kde_kws=dict(cut=3),
                      alpha=.4, edgecolor=(1, 1, 1, .4),
                      ax=ax)

    ax.axvline(x=mean, c="r", ls="--", lw=1)
    ax.axvline(x=mean + std * 3, c="r", ls="--", lw=1)
    ax.axvline(x=mean - std * 3, c="r", ls="--", lw=1)
    ax.set_title(f"{col},mean={mean:0.4f},std={std:0.4f},skew={skew:0.4f},kurt={kurt:0.4f}")
    ax.set_xlabel('')


def plot_ic_qq(df_pl: pl.DataFrame, col: str,
               *,
               ax=None) -> None:
    """IC QQ图

    Examples
    --------
    >>> plot_ic_qq(df_pl, 'RETURN_OO_1')
    """
    a = df_pl[col].to_pandas().dropna()

    sm.qqplot(a, fit=True, line='45', ax=ax)


def plot_ic_heatmap(df_pl: pl.DataFrame, col: str,
                    *,
                    ax=None) -> None:
    """月度IC热力图"""
    df_pl = df_pl.select([_DATE_, col,
                          pl.col(_DATE_).dt.year().alias('year'),
                          pl.col(_DATE_).dt.month().alias('month')
                          ])
    df_pl = df_pl.group_by(by=['year', 'month']).agg(pl.mean(col))
    df_pd = df_pl.to_pandas().set_index(['year', 'month'])

    # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    ax = sns.heatmap(df_pd[col].unstack(), annot=True, cmap='RdYlGn_r', cbar=False, annot_kws={"size": 7}, ax=ax)
    ax.set_title(f"{col},Monthly Mean IC")
    ax.set_xlabel('')


def create_ic_sheet(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str],
                    *,
                    axvlines=()):
    """生成IC图表"""
    df_pl = calc_ic(df_pl, factor, forward_returns)

    for forward_return in forward_returns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        plot_ic_ts(df_pl, forward_return, axvlines=axvlines, ax=axes[0, 0])
        plot_ic_hist(df_pl, forward_return, ax=axes[0, 1])
        plot_ic_qq(df_pl, forward_return, ax=axes[1, 0])
        plot_ic_heatmap(df_pl, forward_return, ax=axes[1, 1])

        fig.tight_layout()
