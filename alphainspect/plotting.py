import math
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels import api as sm

from alphainspect import _QUANTILE_, _DATE_


def get_row_col(count: int):
    """通过图总数，得到二维数量。用于确定合适的子图数量"""
    len_sqrt = math.sqrt(count)
    row, col = math.ceil(len_sqrt), math.floor(len_sqrt)
    if row * col < count:
        col += 1
    return row, col


def plot_heatmap(df: pd.DataFrame,
                 *,
                 title='Mean IC',
                 ax=None) -> None:
    """热力图"""
    # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    ax = sns.heatmap(df, annot=True, cmap='RdYlGn_r', cbar=False, annot_kws={"size": 7}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')


def plot_heatmap_monthly_mean(df: pl.DataFrame, col: str,
                              *,
                              ax=None) -> None:
    """月度平均热力图"""
    df = df.select([_DATE_, col,
                    pl.col(_DATE_).dt.year().alias('year'),
                    pl.col(_DATE_).dt.month().alias('month')
                    ])
    df = df.group_by('year', 'month').agg(pl.mean(col))
    df_pd = df.to_pandas().set_index(['year', 'month'])

    plot_heatmap(df_pd[col].unstack(), title=f"{col},Monthly Mean", ax=ax)


def plot_heatmap_monthly_diff(df: pd.DataFrame, col='G9',
                              *, ax=None) -> None:
    """月度热力图。月底减月初差值

    Parameters
    ----------
    df
    col
    ax

    """
    df = df.select([_DATE_, col,
                    pl.col(_DATE_).dt.year().alias('year'),
                    pl.col(_DATE_).dt.month().alias('month')
                    ]).sort(_DATE_)
    df = df.group_by('year', 'month').agg(pl.last(col) - pl.first(col))
    df_pd = df.to_pandas().set_index(['year', 'month'])

    plot_heatmap(df_pd[col].unstack(), title=f"{col},Monthly Last-First", ax=ax)

    # out = pd.DataFrame(index=df.index)
    # out['year'] = out.index.year
    # out['month'] = out.index.month
    # out['first'] = df[col]
    # out['last'] = df[col]
    # out = out.groupby(by=['year', 'month']).agg({'first': 'first', 'last': 'last'})
    # # 累计收益由累乘改成了累加，这里算法也需要改动
    # # out['cum_ret'] = out['last'] / out['first'] - 1
    # out['cum_ret'] = out['last'] - out['first']
    # plot_heatmap(out['cum_ret'].unstack(), title=f"{col},Monthly Return", ax=ax)


def plot_ts(df: pl.DataFrame, col: str,
            *,
            axvlines=(), ax=None) -> Dict[str, float]:
    """时序图

    Examples
    --------
    >>> plot_ts(df_pd, 'RETURN_OO_1')

    """
    df = df.select([_DATE_, col])

    df = df.select([
        _DATE_,
        pl.col(col),
        pl.col(col).rolling_mean(20).alias('sma_20'),
        pl.col(col).fill_nan(0).cum_sum().alias('cum_sum'),
    ])
    df_pd = df.to_pandas().replace([-np.inf, np.inf], np.nan).dropna(subset=col)
    s: pd.Series = df_pd[col]

    mean = s.mean()
    zscore = s.mean() / s.std(ddof=0)
    ratio = s.abs().gt(0.02).mean()
    t_stat, p_value = stats.ttest_1samp(s, 0)

    title = f"{col},mean={mean:0.4f},zscore={zscore:0.4f}"
    logger.info(title)
    ax1 = df_pd.plot.line(x=_DATE_, y=[col, 'sma_20'], alpha=0.5, lw=1,
                          title=title,
                          ax=ax)
    ax2 = df_pd.plot.line(x=_DATE_, y=['cum_sum'], alpha=0.9, lw=1,
                          secondary_y='cum_sum', c='r',
                          ax=ax1)
    ax1.axhline(y=mean, c="r", ls="--", lw=1)
    ax.set_xlabel('')
    for v in axvlines:
        ax1.axvline(x=v, c="b", ls="--", lw=1)

    return {'mean': mean, 'zscore': zscore, 'ratio': ratio, 't_stat': t_stat, 'p_value': p_value}


def plot_hist(df: pl.DataFrame, col: str,
              *,
              kde: bool = False,
              ax=None) -> Dict[str, float]:
    """直方图

    Parameters
    ----------
    df
    col
        列名
    kde: bool
        是否启用kde。启用后速度慢了非常多
    ax
        子图

    Examples
    --------
    >>> plot_hist(df, 'RETURN_OO_1')
    """
    a = df[col].to_pandas().replace([-np.inf, np.inf], np.nan).dropna()

    mean = a.mean()
    std = a.std(ddof=0)
    skew = a.skew()
    kurt = a.kurt()

    ax = sns.histplot(a,
                      bins=50, kde=kde,
                      stat="density", kde_kws=dict(cut=3),
                      alpha=.4, edgecolor=(1, 1, 1, .4),
                      ax=ax)

    ax.axvline(x=mean, c="r", ls="--", lw=1)
    ax.axvline(x=mean + std * 3, c="r", ls="--", lw=1)
    ax.axvline(x=mean - std * 3, c="r", ls="--", lw=1)
    title = f"{col},std={std:0.4f},skew={skew:0.4f},kurt={kurt:0.4f}"
    logger.info(title)
    ax.set_title(title)
    ax.set_xlabel('')

    return {'std': std, 'skew': skew, 'kurt': kurt, 'mean_': mean}


def plot_qq(df: pl.DataFrame, col: str,
            *,
            ax=None) -> None:
    """QQ图

    Examples
    --------
    >>> plot_qq(df, 'RETURN_OO_1')
    """
    a = df[col].to_pandas().replace([-np.inf, np.inf], np.nan).dropna()

    sm.qqplot(a, fit=True, line='45', ax=ax)


def plot_quantile_bar_mean(df: pl.DataFrame, cols: Sequence[str],
                           *,
                           factor_quantile: str = _QUANTILE_,
                           title: str = 'Mean By Quantile',
                           ax=None):
    """分组平均值柱状图

    Examples
    --------
    >>> plot_quantile_bar_mean(df, ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])

    """
    df = df.group_by(factor_quantile).agg([pl.mean(y) for y in cols]).sort(factor_quantile)
    df_pd = df.to_pandas().set_index(factor_quantile)
    ax = df_pd.plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.bar_label(ax.containers[0])


def plot_quantile_bar_count(df: pl.DataFrame, cols: Sequence[str],
                            *,
                            factor_quantile: str = _QUANTILE_,
                            title: str = 'Count By Quantile',
                            ax=None):
    """分组数量柱状图

    Examples
    --------
    >>> plot_quantile_bar_mean(df, ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])
    """
    df = df.group_by(factor_quantile).agg([pl.count(y) for y in cols]).sort(factor_quantile)
    df_pd = df.to_pandas().set_index(factor_quantile)
    ax = df_pd.plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.bar_label(ax.containers[0])


def plot_quantile_box(df: pl.DataFrame, forward_returns: Sequence[str],
                      *,
                      factor_quantile: str = _QUANTILE_,
                      title: str = 'Box By Quantile',
                      ax=None):
    """分组收益

    Examples
    --------
    >>> plot_quantile_box(df, ['RETURN_OO_1', 'RETURN_OO_2', 'RETURN_CC_1'])

    """
    df = df.select(factor_quantile, *forward_returns)
    df_pd = df.to_pandas().set_index(factor_quantile)

    df_pd = df_pd.stack().reset_index()
    df_pd.columns = ['x', '', 'y']
    df_pd = df_pd.sort_values(by=['x', ''])
    ax = sns.boxplot(data=df_pd, x='x', y='y', hue='', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')


def create_describe1_sheet(df: pl.DataFrame, cols: Sequence[str], factor_quantile: str = _QUANTILE_):
    """单因子分组统计

    Parameters
    ----------
    df
    cols
    factor_quantile

    Returns
    -------

    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # 一定要过滤null才能用
    df = df.filter(pl.col(factor_quantile).is_not_null())

    plot_quantile_bar_count(df, cols, factor_quantile=factor_quantile, ax=axes[0])
    plot_quantile_bar_mean(df, cols, factor_quantile=factor_quantile, ax=axes[1])
    plot_quantile_box(df, cols, factor_quantile=factor_quantile, ax=axes[2])

    fig.tight_layout()


def create_describe2_sheet(df: pl.DataFrame,
                           col: str,
                           factor_quantiles: Sequence[str]):
    """双因子分组统计。灵活使用分组方法能实现独立双重排序和条件双重排序

    例如，将两个因子划分成3*5，查看两因子组合效果

    Parameters
    ----------
    df
    col
        可以是收益率，也可以是其他需要统计的值
    factor_quantiles

    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 9))

    df = df.filter([pl.col(q).is_not_null() for q in factor_quantiles])

    df_mean = df.group_by(*factor_quantiles).agg(pl.mean(col)).sort(*factor_quantiles).to_pandas()
    df_std = df.group_by(*factor_quantiles).agg(pl.std(col, ddof=0)).sort(*factor_quantiles).to_pandas()
    df_count = df.group_by(*factor_quantiles).agg(pl.count(col)).sort(*factor_quantiles).to_pandas()

    df_mean = df_mean.set_index(factor_quantiles)[col].unstack()
    df_std = df_std.set_index(factor_quantiles)[col].unstack()
    df_count = df_count.set_index(factor_quantiles)[col].unstack()

    plot_heatmap(df_mean, title='Mean', ax=axes[0])
    plot_heatmap(df_std, title='Std', ax=axes[1])
    plot_heatmap(df_count, title='Count', ax=axes[2])

    fig.tight_layout()
