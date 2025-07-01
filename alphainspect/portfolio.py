from typing import Tuple

import polars as pl
import polars.selectors as cs
from loguru import logger
from matplotlib import pyplot as plt

from alphainspect import _QUANTILE_, _DATE_
from alphainspect.plotting import plot_heatmap_monthly_diff


def _points_9(df: pl.DataFrame):
    """通过几个数字得到疏密情况

    Returns
    -------
    H,M,L
        前三数表示高中低区最大层(852)相对平均距离的倍数
    HL/ML
      高区比中区大小
    AVG
        平均距离

    """
    N0 = df[-1, "0"]
    N1 = df[-1, "1"]
    N2 = df[-1, "2"]
    N3 = df[-1, "3"]
    N4 = df[-1, "4"]
    N5 = df[-1, "5"]
    N6 = df[-1, "6"]
    N7 = df[-1, "7"]
    N8 = df[-1, "8"]

    # 几何平均
    avg = (abs(N7 - N6) * abs(N4 - N3) * abs(N1 - N0)) ** (1 / 3)

    # 多头部分。2表示均匀，1表示中间线与上重合，0表示中间线与下重合
    H = (N8 - N6) / avg  # 76顺序可换
    M = (N5 - N3) / avg  # 43顺序可换
    L = (N2 - N0) / avg  # 21顺序可换
    # 三部分统一成一起，继续计算
    A = (N8 + N7 + N6 - N2 - N1 - N0) / (N5 + N4 + N3 - N2 - N1 - N0)
    return round(H, 1), round(M, 1), round(L, 1), round(A, 2), round(avg, 3)


def _points_6(df: pl.DataFrame):
    N0 = df[-1, "0"]
    N1 = df[-1, "1"]
    N2 = df[-1, "2"]
    N3 = df[-1, "3"]
    N4 = df[-1, "4"]
    N5 = df[-1, "5"]

    # 几何平均
    avg = (abs(N5 - N4) * abs(N3 - N2) * abs(N1 - N0)) ** (1 / 3)

    # 多头部分。2表示均匀，1表示中间线与上重合，0表示中间线与下重合
    H = (N5 - N4) / avg
    M = (N3 - N2) / avg
    L = (N1 - N0) / avg
    # 三部分统一成一起，继续计算
    A = (N5 + N4 - N1 - N0) / (N3 + N2 - N1 - N0)
    return round(H, 1), round(M, 1), round(L, 1), round(A, 2), round(avg, 3)


def _points_4(df: pl.DataFrame):
    N0 = df[-1, "0"]
    N1 = df[-1, "1"]
    N2 = df[-1, "2"]
    N3 = df[-1, "3"]

    # 几何平均
    avg = (abs(N3 - N2) * abs(N2 - N1) * abs(N1 - N0)) ** (1 / 3)

    # 多头部分。2表示均匀，1表示中间线与上重合，0表示中间线与下重合
    H = (N3 - N2) / avg
    M = (N2 - N1) / avg
    L = (N1 - N0) / avg
    # 三部分统一成一起，继续计算
    A = (N3 + N2 - N1 - N0) / (N2 + N1 - N1 - N0)
    return round(H, 1), round(M, 1), round(L, 1), round(A, 2), round(avg, 3)


def points(df: pl.DataFrame):
    count = len(df.columns)
    if count == 9 + 2:
        return _points_9(df)
    if count == 6 + 2:
        return _points_6(df)
    if count == 4 + 2:
        return _points_4(df)
    return None


def calc_cum_return_by_quantile(df: pl.DataFrame, fwd_ret_1: str, factor_quantile: str = _QUANTILE_) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """分层计算收益。分成N层，层内等权

    单利不加仓。不考虑手续费，不考虑资金不足，资金也不分多份。每天入场每天出场。
    此项目只为了评价因子是否有效，在累计收益计算时是否精确并不重要，这里的目的只是为了分层清晰，更精确计算请用其它工具

    在输入收益率时，即可以输入日频收益率，也可以输入几何平均后的收益率。

    """
    x = df.filter(pl.col(factor_quantile) >= 0).group_by(factor_quantile, _DATE_).agg(pl.mean(fwd_ret_1))
    y = x.pivot(index=_DATE_, columns=factor_quantile, values=fwd_ret_1, aggregate_function='first', sort_columns=True).sort(_DATE_)
    ret = y.with_columns(cs.numeric().fill_nan(None))
    cum = ret.with_columns(long_short=cs.by_index(-1).as_expr() - cs.by_index(1).as_expr()).with_columns(cs.numeric().fill_null(0).cum_sum())
    avg = ret.select(_DATE_, cs.numeric().mean())
    std = ret.select(_DATE_, cs.numeric().std(ddof=0))

    return ret, cum, avg, std


def plot_quantile_portfolio(df: pl.DataFrame,
                            fwd_ret_1: str,
                            long_short: str = 'long_short',
                            *,
                            show_long_short: bool = True,
                            axvlines=None, ax=None) -> None:
    try:
        logger.info("{},{}", fwd_ret_1, points(df))
    except Exception:
        logger.info("{}", df.columns)
    df_pd = df.to_pandas().set_index(_DATE_)
    if long_short is None:
        ax = df_pd.plot(ax=ax, title=f'{fwd_ret_1}', cmap='coolwarm', lw=1, grid=True)
    else:
        ax = df_pd.drop(columns=long_short).plot(ax=ax, title=f'{fwd_ret_1}', cmap='coolwarm', lw=1, grid=True)
        if show_long_short:
            df_pd[long_short].plot(ax=ax, c="g", ls="--", lw=1, label='L-S', grid=True)
    ax.legend(loc='upper left')
    ax.set_xlabel('')
    for v in axvlines:
        ax.axvline(x=v, c="b", ls="--", lw=1)


def create_portfolio_sheet(df: pl.DataFrame,
                           fwd_ret_1: str,
                           factor_quantile: str = _QUANTILE_,
                           *,
                           axvlines=()) -> None:
    """分层累计收益图"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    # 分层累计收益
    ret, cum, avg, std = calc_cum_return_by_quantile(df, fwd_ret_1, factor_quantile)
    plot_quantile_portfolio(cum, fwd_ret_1, axvlines=axvlines, ax=axes[0])
    groups = cum.columns[1], cum.columns[-2]
    for i, g in enumerate(groups):
        ax = plt.subplot(223 + i)
        # 月度收益
        plot_heatmap_monthly_diff(cum, g, ax=ax)
    fig.tight_layout()
