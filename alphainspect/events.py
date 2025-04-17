from functools import lru_cache
from typing import Sequence, List, Optional

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

from alphainspect import _QUANTILE_, _DATE_, _ASSET_
from alphainspect.portfolio import calc_cum_return_by_quantile, plot_quantile_portfolio

# 以+-开头的纯数字，希望不会与其他列名冲突
_REG_AROUND_ = r'^[+-]\d+$'
_COL_AROUND_ = pl.col(_REG_AROUND_)


@lru_cache
def make_around_columns(periods_before: int = 3, periods_after: int = 15) -> List[str]:
    """生成表格区表头"""
    return [f'{i:+02d}' for i in range(-periods_before, periods_after + 1)]


def with_around_price(df: pl.DataFrame, price: str, periods_before: int = 5, periods_after: int = 15) -> pl.DataFrame:
    """添加前后复权价

    Parameters
    ----------
    df
    price
        收盘价或均价
    periods_before
    periods_after

    """

    def _func_ts(df: pl.DataFrame,
                 normalize: bool = True):
        # 一定要排序
        df = df.sort(_DATE_)
        n = len(df)

        t0 = df[price].to_numpy()
        # 准备数据，前后要留空间
        a = np.empty(n + periods_before + periods_after, dtype=t0.dtype)
        a[:periods_before] = np.nan
        a[-periods_after - 1:] = np.nan
        a[periods_before:periods_before + n] = t0

        # 滑动窗口
        b = sliding_window_view(a, periods_before + periods_after + 1)
        # 将T+0置为1
        if normalize:
            b = b / b[:, [periods_before]]
        # numpy转polars
        c = pl.from_numpy(b, schema=make_around_columns(periods_before, periods_after))
        return df.with_columns(c)

    return df.group_by(_ASSET_).map_groups(_func_ts).with_columns(_COL_AROUND_.fill_nan(None))


def plot_events_errorbar(df: pl.DataFrame, factor_quantile: str = _QUANTILE_, ax=None) -> None:
    """事件前后误差条。只显示最大分组"""
    min_max = df.select(pl.min(factor_quantile).alias('min'), pl.max(factor_quantile).alias('max'))
    min_max = min_max.to_dicts()[0]
    _min, _max = min_max['min'], min_max['max']

    df = df.select(factor_quantile, _COL_AROUND_)
    mean_pl = df.group_by(factor_quantile).agg(pl.mean(_REG_AROUND_)).sort(factor_quantile)
    mean_pd: pd.DataFrame = mean_pl.to_pandas().set_index(factor_quantile).T
    std_pl = df.group_by(factor_quantile).agg(pl.std(_REG_AROUND_)).sort(factor_quantile)
    std_pd: pd.DataFrame = std_pl.to_pandas().set_index(factor_quantile).T

    # 取最大分组
    a = mean_pd.loc[:, _max]
    b = std_pd.loc[:, _max]

    ax.errorbar(x=a.index, y=a, yerr=b)
    ax.axvline(x=a.index.get_loc('+0'), c="r", ls="--", lw=1)
    ax.set_xlabel('')
    ax.set_title(f'Quantile {_max} errorbar')


def plot_events_average(df: pl.DataFrame, factor_quantile: str = _QUANTILE_, ax=None) -> None:
    """事件前后标准化后平均价"""
    df = df.select(factor_quantile, _COL_AROUND_)
    mean_pl = df.group_by(factor_quantile).agg(pl.mean(_REG_AROUND_)).sort(factor_quantile)
    mean_pd: pd.DataFrame = mean_pl.to_pandas().set_index(factor_quantile).T
    mean_pd.plot.line(title='Average Cumulative Returns by Quantile', ax=ax, cmap='coolwarm', lw=1)
    ax.axvline(x=mean_pd.index.get_loc('+0'), c="r", ls="--", lw=1)
    ax.set_xlabel('')


def plot_events_count(df: pl.DataFrame, axvlines: Sequence[str] = (), ax=None) -> None:
    """事件发生次数"""
    df = df.group_by(_DATE_).count().sort(_DATE_)
    df_pd = df.to_pandas().set_index(_DATE_)
    df_pd.plot.line(title='Distribution of events', ax=ax, lw=1, grid=True)
    ax.set_xlabel('')
    for v in axvlines:
        ax.axvline(x=v, c="b", ls="--", lw=1)


def plot_events_ratio(df: pl.DataFrame, fwd_ret_1: str, factor_quantile: str = _QUANTILE_, axvlines: Sequence[str] = (), ax=None) -> None:
    """事件胜率"""
    df = df.group_by(_DATE_, factor_quantile).agg((pl.col(fwd_ret_1) > 0).mean()).sort(_DATE_)
    df_pd = df.to_pandas().set_index([_DATE_, factor_quantile])[fwd_ret_1].unstack()
    df_pd.plot.line(title='Win Ratio of events', ax=ax, lw=1, grid=True)
    ax.set_xlabel('')
    for v in axvlines:
        ax.axvline(x=v, c="b", ls="--", lw=1)


def create_events_sheet(df: pl.DataFrame,
                        condition: Optional[pl.Expr],
                        fwd_ret_1: str,
                        show_long_short: bool = True,
                        factor_quantile: str = _QUANTILE_, axvlines: Sequence[str] = ()):
    """事件分析图表

    Parameters
    ----------
    df
    condition
        条件，分析前先过滤。
    fwd_ret_1:str
        用于记算累计收益的1期远期收益率
    show_long_short:bool
        是否显示多空对冲收益
    factor_quantile:str
        分层。可以一层，也可以多层。
    axvlines

    """
    # 可在外部提前过滤
    if condition is not None:
        df = df.filter(condition)
    # 一定要过滤空值
    df = df.filter(pl.col(factor_quantile).is_not_null())

    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    axes = axes.flatten()

    plot_events_average(df, factor_quantile=factor_quantile, ax=axes[0])
    plot_events_errorbar(df, factor_quantile=factor_quantile, ax=axes[1])
    plot_events_ratio(df, fwd_ret_1, factor_quantile=factor_quantile, axvlines=axvlines, ax=axes[2])
    # 画累计收益
    ret, cum, avg, std = calc_cum_return_by_quantile(df, fwd_ret_1, factor_quantile)
    plot_quantile_portfolio(cum, fwd_ret_1, show_long_short=show_long_short, axvlines=axvlines, ax=axes[3])
    plot_events_count(df, axvlines=axvlines, ax=axes[4])

    fig.tight_layout()
