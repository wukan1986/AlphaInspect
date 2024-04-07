import itertools
from typing import Sequence, Literal, Dict

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from matplotlib import pyplot as plt
from polars import Expr
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from statsmodels import api as sm

from alphainspect import _DATE_
from alphainspect.utils import plot_heatmap, get_row_col, select_by_suffix, plot_hist


def rank_ic(a: str, b: str) -> Expr:
    """RankIC"""
    return pl.corr(a, b, method='spearman', ddof=0, propagate_nans=False)


def mutual_info(a: str, b: str) -> Expr:
    """互信息"""

    def mutual_info_func(xx) -> float:
        yx = np.vstack(xx).T
        # 跳过nan
        mask = np.any(np.isnan(yx), axis=1)
        yx_ = yx[~mask, :]
        if len(yx_) <= 3:
            return np.nan
        # TODO 使用此函数是否合理？
        mi = mutual_info_regression(yx_[:, 0].reshape(-1, 1), yx_[:, 1], n_neighbors=3)
        return float(mi[0])

    return pl.map_groups([a, b], lambda xx: mutual_info_func(xx))


def w_corr(a: str, b: str, w: str) -> pl.Expr:
    def _w_corr(xx):
        x, y, weights = xx
        cov_matrix = np.cov(x, y, aweights=weights)
        weighted_corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
        return weighted_corr

    return pl.map_groups([a, b, w], lambda xx: _w_corr(xx))


def calc_ic(df_pl: pl.DataFrame, factors: Sequence[str], forward_returns: Sequence[str],
            method: Literal['rank_ic', 'mutual_info'] = 'rank_ic') -> pl.DataFrame:
    """多因子多收益的IC矩阵。方便部分用户统计大量因子信息"""
    if method == 'mutual_info':
        # 互信息，非线性因子。注意，有点慢
        func = mutual_info
    else:
        # RankIC，线性因子
        func = rank_ic

    return df_pl.group_by(_DATE_).agg(
        [func(x, y).alias(f'{x}__{y}') for x, y in itertools.product(factors, forward_returns)]
    ).sort(_DATE_).fill_nan(None)


def calc_ic_mean(df_pl: pl.DataFrame) -> pl.DataFrame:
    """计算IC的均值"""
    return df_pl.select(pl.exclude(_DATE_).mean())


def calc_ic_ir(df_pl: pl.DataFrame) -> pl.DataFrame:
    """计算ir,需保证没有nan，只有null"""
    return df_pl.select(pl.exclude(_DATE_).mean() / pl.exclude(_DATE_).std(ddof=0))


def calc_ic_corr(df_pl: pl.DataFrame) -> pd.DataFrame:
    """由于numpy版不能很好的处理空值，所以用pandas版"""
    return df_pl.to_pandas().corr(method="pearson")


def row_unstack(df_pl: pl.DataFrame, factors: Sequence[str], forward_returns: Sequence[str]) -> pd.DataFrame:
    """一行值堆叠成一个矩阵"""
    return pd.DataFrame(df_pl.to_numpy().reshape(len(factors), len(forward_returns)),
                        index=factors, columns=forward_returns)


def plot_ic_ts(df_pl: pl.DataFrame, col: str,
               *,
               axvlines=(), ax=None) -> Dict[str, float]:
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
    ratio = (s.abs() > 0.02).mean()
    t_stat, p_value = stats.ttest_1samp(s, 0)

    title = f"{col},IC={ic:0.4f},>0.02={ratio:0.2f},IR={ir:0.4f},t_stat={t_stat:0.4f},p_value={p_value:0.4f}"
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

    return {'IC': ic, 'IR': ir, 'ratio': ratio, 't_stat': t_stat, 'p_value': p_value}


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


def plot_ic_heatmap_monthly(df_pl: pl.DataFrame, col: str,
                            *,
                            ax=None) -> None:
    """月度IC热力图"""
    df_pl = df_pl.select([_DATE_, col,
                          pl.col(_DATE_).dt.year().alias('year'),
                          pl.col(_DATE_).dt.month().alias('month')
                          ])
    df_pl = df_pl.group_by('year', 'month').agg(pl.mean(col))
    df_pd = df_pl.to_pandas().set_index(['year', 'month'])

    plot_heatmap(df_pd[col].unstack(), title=f"{col},Monthly Mean IC", ax=ax)


def create_ic1_sheet(df_pl: pl.DataFrame, factor: str, forward_returns: Sequence[str],
                     *,
                     axvlines=(),
                     method: Literal['rank_ic', 'mutual_info'] = 'rank_ic') -> pl.DataFrame:
    """生成IC图表系列。单因子多收益率"""
    df_pl = calc_ic(df_pl, [factor], forward_returns, method)

    for col in df_pl.columns:
        if col == _DATE_:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        plot_ic_ts(df_pl, col, axvlines=axvlines, ax=axes[0, 0])
        plot_hist(df_pl, col, ax=axes[0, 1])
        plot_ic_qq(df_pl, col, ax=axes[1, 0])
        plot_ic_heatmap_monthly(df_pl, col, ax=axes[1, 1])

        fig.tight_layout()

    return df_pl


def create_ic2_sheet(df_pl: pl.DataFrame, factors: Sequence[str], forward_returns: Sequence[str],
                     *,
                     axvlines=(), ) -> pl.DataFrame:
    """生成IC图表。多因子多收益率。

    用于分析相似因子在不同持有期下的IC信息
    """
    df_pl = calc_ic(df_pl, factors, forward_returns)
    df_ic = calc_ic_mean(df_pl)
    df_ir = calc_ic_ir(df_pl)
    df_ic = row_unstack(df_ic, factors, forward_returns)
    df_ir = row_unstack(df_ir, factors, forward_returns)
    logger.info('Mean IC: {} \n{}', '=' * 60, df_ic)
    logger.info('IC_IR: {} \n{}', '=' * 60, df_ir)

    # 画ic与ir的热力图
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    plot_heatmap(df_ic, title='Mean IC', ax=axes[0])
    plot_heatmap(df_ir, title='IR', ax=axes[1])
    fig.tight_layout()

    # IC之间相关性，可用于检查多重共线性
    corrs = {}
    for forward in forward_returns:
        corrs[forward] = calc_ic_corr(select_by_suffix(df_pl, f'__{forward}'))

    row, col = get_row_col(len(corrs))
    fig, axes = plt.subplots(row, col, figsize=(12, 9), squeeze=False)
    axes = axes.flatten()
    for i, (k, v) in enumerate(corrs.items()):
        plot_heatmap(v, title=f'{k} IC Corr', ax=axes[i])
    fig.tight_layout()

    # 画ic时序图
    fig, axes = plt.subplots(len(factors), len(forward_returns), figsize=(12, 9), squeeze=False)
    axes = axes.flatten()
    logger.info('IC TimeSeries: {}', '=' * 60)
    for i, (x, y) in enumerate(itertools.product(factors, forward_returns)):
        plot_ic_ts(df_pl, f'{x}__{y}', axvlines=axvlines, ax=axes[i])
    fig.tight_layout()

    return df_pl
