import itertools
from typing import Sequence, Literal

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from polars import Expr
from sklearn.feature_selection import mutual_info_regression
from statsmodels import api as sm

from alphainspect import _DATE_


def rank_ic(a: str, b: str) -> Expr:
    """RankIC"""
    return pl.corr(a, b, method='spearman', ddof=0, propagate_nans=False)


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
    return df_pl.group_by(_DATE_).agg(
        # 这里没有换名，名字将与forward_returns对应
        [rank_ic(x, factor) for x in forward_returns]
    ).sort(_DATE_).fill_nan(None)


def calc_ic_mean(df_pl: pl.DataFrame):
    return df_pl.select(pl.exclude(_DATE_).mean())


def calc_ic_ir(df_pl: pl.DataFrame):
    """计算ir,需保证没有nan，只有null"""
    return df_pl.select(pl.exclude(_DATE_).mean() / pl.exclude(_DATE_).std(ddof=0))


def row_unstack(df_pl: pl.DataFrame, factors: Sequence[str], forward_returns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(df_pl.to_numpy().reshape(len(factors), len(forward_returns)),
                        index=factors, columns=forward_returns)


def mutual_info_func(xx):
    yx = np.vstack(xx).T
    # 跳过nan
    mask = np.any(np.isnan(yx), axis=1)
    yx_ = yx[~mask, :]
    if len(yx_) <= 3:
        return np.nan
    # TODO 使用此函数是否合理？
    mi = mutual_info_regression(yx_[:, 0].reshape(-1, 1), yx_[:, 1], n_neighbors=3)
    return float(mi[0])


def mutual_info(a: str, b: str) -> Expr:
    """mutual_info"""
    return pl.map_groups([a, b], lambda xx: mutual_info_func(xx))


def calc_mi(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str]) -> pl.DataFrame:
    return df_pl.group_by(_DATE_).agg(
        # 这里没有换名，名字将与forward_returns对应
        [mutual_info(x, factor) for x in forward_returns]
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
    df_pd = df_pl.to_pandas().replace([-np.inf, np.inf], np.nan).dropna(subset='ic')
    s: pd.Series = df_pd['ic']

    ic = s.mean()
    ir = s.mean() / s.std()
    rate = (s.abs() > 0.02).value_counts(normalize=True).loc[True]

    title = f"{col},IC={ic:0.4f},>0.02={rate:0.2f},IR={ir:0.4f}"
    logger.info(title)
    ax1 = df_pd.plot.line(x=_DATE_, y=['ic', 'sma_20'], alpha=0.5, lw=1,
                          title=title,
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
    a = df_pl[col].to_pandas().replace([-np.inf, np.inf], np.nan).dropna()

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
    title = f"{col},mean={mean:0.4f},std={std:0.4f},skew={skew:0.4f},kurt={kurt:0.4f}"
    logger.info(title)
    ax.set_title(title)
    ax.set_xlabel('')


def plot_ic_qq(df_pl: pl.DataFrame, col: str,
               *,
               ax=None) -> None:
    """IC QQ图

    Examples
    --------
    >>> plot_ic_qq(df_pl, 'RETURN_OO_1')
    """
    a = df_pl[col].to_pandas().replace([-np.inf, np.inf], np.nan).dropna()

    sm.qqplot(a, fit=True, line='45', ax=ax)


def plot_ic_heatmap(df_pl: pl.DataFrame, col: str,
                    *,
                    ax=None) -> None:
    """月度IC热力图"""
    df_pl = df_pl.select([_DATE_, col,
                          pl.col(_DATE_).dt.year().alias('year'),
                          pl.col(_DATE_).dt.month().alias('month')
                          ])
    df_pl = df_pl.group_by('year', 'month').agg(pl.mean(col))
    df_pd = df_pl.to_pandas().set_index(['year', 'month'])

    # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    ax = sns.heatmap(df_pd[col].unstack(), annot=True, cmap='RdYlGn_r', cbar=False, annot_kws={"size": 7}, ax=ax)
    ax.set_title(f"{col},Monthly Mean IC")
    ax.set_xlabel('')


def create_ic_sheet(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str],
                    *,
                    axvlines=(),
                    method: Literal['rank_ic', 'mutual_info'] = 'rank_ic'):
    """生成IC图表"""
    if method == 'mutual_info':
        # 互信息，非线性因子。注意，有点慢
        df_pl = calc_mi(df_pl, factor, forward_returns)
    else:
        # RankIC，线性因子
        df_pl = calc_ic(df_pl, factor, forward_returns)

    for forward_return in forward_returns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        plot_ic_ts(df_pl, forward_return, axvlines=axvlines, ax=axes[0, 0])
        plot_ic_hist(df_pl, forward_return, ax=axes[0, 1])
        plot_ic_qq(df_pl, forward_return, ax=axes[1, 0])
        plot_ic_heatmap(df_pl, forward_return, ax=axes[1, 1])

        fig.tight_layout()


def calc_ic2(df_pl: pl.DataFrame, factors: Sequence[str], forward_returns: Sequence[str]) -> pl.DataFrame:
    """多因子多收益的IC矩阵。方便部分用户统计大量因子信息"""
    return df_pl.group_by(_DATE_).agg(
        [rank_ic(x, y).alias(f'{x}__{y}') for x, y in itertools.product(factors, forward_returns)]
    ).sort(_DATE_).fill_nan(None)


def plot_ic2_heatmap(df_pd: pd.DataFrame,
                     *,
                     title='Mean IC',
                     ax=None) -> None:
    """多个IC的热力图"""
    ax = sns.heatmap(df_pd, annot=True, cmap='RdYlGn_r', cbar=False, annot_kws={"size": 7}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')


def create_ic2_sheet(df_pl: pl.DataFrame, factors: Sequence[str], forward_returns: Sequence[str],
                     *,
                     axvlines=(), ):
    df_pl = calc_ic2(df_pl, factors, forward_returns)
    df_ic = calc_ic_mean(df_pl)
    df_ir = calc_ic_ir(df_pl)
    df_ic = row_unstack(df_ic, factors, forward_returns)
    df_ir = row_unstack(df_ir, factors, forward_returns)
    logger.info('Mean IC: {} \n{}', '=' * 60, df_ic)
    logger.info('IC_IR: {} \n{}', '=' * 60, df_ir)

    # 画ic与ir的热力图
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    plot_ic2_heatmap(df_ic, title='Mean IC', ax=axes[0])
    plot_ic2_heatmap(df_ir, title='IR', ax=axes[1])
    fig.tight_layout()

    # 画ic时序图
    fig, axes = plt.subplots(len(factors), len(forward_returns), figsize=(12, 9))
    axes = axes.flatten()
    logger.info('IC TimeSeries: {}', '=' * 60)
    for i, (x, y) in enumerate(itertools.product(factors, forward_returns)):
        plot_ic_ts(df_pl, f'{x}__{y}', axvlines=axvlines, ax=axes[i])
    fig.tight_layout()
