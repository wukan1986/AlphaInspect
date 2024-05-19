from functools import lru_cache
from typing import Sequence, List

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

from alphainspect import _QUANTILE_, _DATE_, _ASSET_

_REG_AROUND_ = r'^[+-]\d+$'
_COL_AROUND_ = pl.col(_REG_AROUND_)


@lru_cache
def make_around_columns(periods_before: int = 3, periods_after: int = 15) -> List[str]:
    """生成表格区表头"""
    return [f'{i:+02d}' for i in range(-periods_before, periods_after + 1)]


def with_around_price(df_pl: pl.DataFrame, price: str, periods_before: int = 5, periods_after: int = 15) -> pl.DataFrame:
    """添加事件前后复权价

    Parameters
    ----------
    df_pl
    price
    periods_before
    periods_after

    Returns
    -------

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

    return df_pl.group_by(_ASSET_).map_groups(_func_ts).with_columns(_COL_AROUND_.fill_nan(None))


def plot_events_errorbar(df_pl: pl.DataFrame, factor_quantile: str = _QUANTILE_, ax=None) -> None:
    """事件前后误差条"""
    min_max = df_pl.select(pl.min(factor_quantile).alias('min'), pl.max(factor_quantile).alias('max'))
    min_max = min_max.to_dicts()[0]
    _min, _max = min_max['min'], min_max['max']

    df_pl = df_pl.select(factor_quantile, _COL_AROUND_)
    mean_pl = df_pl.group_by(factor_quantile).agg(pl.mean(_REG_AROUND_)).sort(factor_quantile)
    mean_pd: pd.DataFrame = mean_pl.to_pandas().set_index(factor_quantile).T
    std_pl = df_pl.group_by(factor_quantile).agg(pl.std(_REG_AROUND_)).sort(factor_quantile)
    std_pd: pd.DataFrame = std_pl.to_pandas().set_index(factor_quantile).T

    a = mean_pd.loc[:, _max]
    b = std_pd.loc[:, _max]

    ax.errorbar(x=a.index, y=a, yerr=b)
    ax.axvline(x=a.index.get_loc('+0'), c="r", ls="--", lw=1)
    ax.set_xlabel('')
    ax.set_title(f'Quantile {_max} errorbar')


def plot_events_average(df_pl: pl.DataFrame, factor_quantile: str = _QUANTILE_, ax=None) -> None:
    """事件前后标准化后平均价"""
    df_pl = df_pl.select(factor_quantile, _COL_AROUND_)
    mean_pl = df_pl.group_by(factor_quantile).agg(pl.mean(_REG_AROUND_)).sort(factor_quantile)
    mean_pd: pd.DataFrame = mean_pl.to_pandas().set_index(factor_quantile).T
    mean_pd.plot.line(title='Average Cumulative Returns by Quantile', ax=ax, cmap='coolwarm', lw=1)
    ax.axvline(x=mean_pd.index.get_loc('+0'), c="r", ls="--", lw=1)
    ax.set_xlabel('')


def plot_events_count(df_pl: pl.DataFrame, axvlines: Sequence[str] = (), ax=None) -> None:
    """事件发生次数"""
    df_pl = df_pl.group_by(_DATE_).count().sort(_DATE_)
    df_pd = df_pl.to_pandas().set_index(_DATE_)
    df_pd.plot.line(title='Distribution of events', ax=ax, lw=1, grid=True)
    ax.set_xlabel('')
    for v in axvlines:
        ax.axvline(x=v, c="b", ls="--", lw=1)


def create_events_sheet(df_pl: pl.DataFrame, condition: pl.Expr, factor_quantile: str = _QUANTILE_, axvlines: Sequence[str] = ()):
    # 一定要过滤空值
    df_pl = df_pl.filter(pl.col(factor_quantile).is_not_null()).filter(condition)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12))

    plot_events_count(df_pl, ax=axes[0], axvlines=axvlines)
    plot_events_average(df_pl, factor_quantile=factor_quantile, ax=axes[1])
    plot_events_errorbar(df_pl, factor_quantile=factor_quantile, ax=axes[2])

    fig.tight_layout()
