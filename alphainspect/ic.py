import itertools
from typing import Sequence, Literal

import numpy as np
import polars as pl
from loguru import logger
from matplotlib import pyplot as plt
from polars import Expr
from sklearn.feature_selection import mutual_info_regression

from alphainspect import _DATE_
from alphainspect.calc import calc_mean, calc_ir, calc_corr
from alphainspect.plotting import plot_heatmap, get_row_col, plot_hist, plot_heatmap_monthly_mean, plot_qq, plot_ts
from alphainspect.utils import select_by_suffix, index_split_unstack


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


def calc_ic(df: pl.DataFrame, factors: Sequence[str], forward_returns: Sequence[str],
            method: Literal['rank_ic', 'mutual_info'] = 'rank_ic') -> pl.DataFrame:
    """多因子多收益的IC矩阵。方便部分用户统计大量因子信息"""
    if method == 'mutual_info':
        # 互信息，非线性因子。注意，有点慢
        func = mutual_info
    else:
        # RankIC，线性因子
        func = rank_ic

    return df.group_by(_DATE_).agg(
        [func(x, y).alias(f'{x}__{y}') for x, y in itertools.product(factors, forward_returns)]
    ).sort(_DATE_).fill_nan(None)


def create_ic1_sheet(df: pl.DataFrame, factor: str, forward_returns: Sequence[str],
                     *,
                     axvlines=(),
                     method: Literal['rank_ic', 'mutual_info'] = 'rank_ic') -> pl.DataFrame:
    """生成IC图表系列。单因子多收益率"""
    df = calc_ic(df, [factor], forward_returns, method)

    for col in df.columns:
        if col == _DATE_:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        plot_ts(df, col, axvlines=axvlines, ax=axes[0, 0])
        plot_hist(df, col, ax=axes[0, 1])
        plot_qq(df, col, ax=axes[1, 0])
        plot_heatmap_monthly_mean(df, col, ax=axes[1, 1])

        fig.tight_layout()

    return df


def create_ic2_sheet(df: pl.DataFrame, factors: Sequence[str], forward_returns: Sequence[str],
                     *,
                     axvlines=(), ) -> pl.DataFrame:
    """生成IC图表。多因子多收益率。

    用于分析相似因子在不同持有期下的IC信息
    """
    df = calc_ic(df, factors, forward_returns)
    df_ic = calc_mean(df).to_pandas().iloc[0]
    df_ir = calc_ir(df).to_pandas().iloc[0]

    df_ic = index_split_unstack(df_ic)
    df_ir = index_split_unstack(df_ir)

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
        corrs[forward] = calc_corr(select_by_suffix(df, f'__{forward}'))

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
        plot_ts(df, f'{x}__{y}', axvlines=axvlines, ax=axes[i])
    fig.tight_layout()

    return df
