from typing import Tuple

import polars as pl
import polars.selectors as cs
from matplotlib import pyplot as plt

from alphainspect import _QUANTILE_, _DATE_
from alphainspect.plotting import plot_heatmap_monthly_diff


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

    # avg = ret.select(cs.numeric().mean()).to_pandas().iloc[0]
    # std = ret.select(cs.numeric().std(ddof=0)).to_pandas().iloc[0]
    # ret = ret.to_pandas().set_index(_DATE_)
    # return ret, cum, avg, std
    avg = ret.select(_DATE_, cs.numeric().mean())
    std = ret.select(_DATE_, cs.numeric().std(ddof=0))
    return ret, cum, avg, std


def plot_quantile_portfolio(df: pl.DataFrame,
                            fwd_ret_1: str,
                            long_short: str = 'long_short',
                            *,
                            show_long_short: bool = True,
                            axvlines=None, ax=None) -> None:
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
