"""
废除
"""
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from matplotlib import pyplot as plt

from alphainspect import _QUANTILE_, _DATE_, _ASSET_, _WEIGHT_
from alphainspect.portfolio import plot_quantile_portfolio
from alphainspect.utils import cumulative_returns


def _calc_cum_return_by_quantile(df: pl.DataFrame, fwd_ret_1: str, period: int = 5, factor_quantile: str = _QUANTILE_) -> pd.DataFrame:
    """分层计算收益。分成N层，层内等权"""
    q_max = df.select(pl.max(factor_quantile)).to_series(0)[0]
    rr = df.pivot(index=_DATE_, columns=_ASSET_, values=fwd_ret_1, aggregate_function='first', sort_columns=True).sort(_DATE_)
    qq = df.pivot(index=_DATE_, columns=_ASSET_, values=factor_quantile, aggregate_function='first', sort_columns=True).sort(_DATE_)

    out = pd.DataFrame(index=rr[_DATE_])
    rr = rr.select(pl.exclude(_DATE_)).to_numpy() + 1  # 日收益
    qq = qq.select(pl.exclude(_DATE_)).to_numpy()  # 分组编号
    # logger.info('累计收益准备数据,period={}', period)

    np.seterr(divide='ignore', invalid='ignore')
    for i in range(int(q_max) + 1):
        # 等权
        b = qq == i
        w = b / b.sum(axis=1).reshape(-1, 1)
        w[(w == 0).all(axis=1), :] = np.nan
        # 权重绝对值和为1
        out[f'G{i}'] = cumulative_returns(rr, w, funds=period, freq=period)
    # !!!直接减是错误的，因为两资金是独立的，资金减少的一份由于资金不足对冲比例已经不再是1:1
    # out['spread'] = out[f'G{q_max}'] - out[f'G0']
    logger.info('累计收益计算完成,period={}\n{}', period, out.tail(1).to_string())
    return out


def calc_cum_return_spread(df: pl.DataFrame, fwd_ret_1: str, period: int = 5, factor_quantile: str = _QUANTILE_) -> pd.DataFrame:
    """分层计算收益。分成N层，层内等权。
    取Top层和Bottom层。比较不同的计算方法多空收益的区别"""

    q_max = df.select(pl.max(factor_quantile)).to_series(0)[0]
    rr = df.pivot(index=_DATE_, columns=_ASSET_, values=fwd_ret_1, aggregate_function='first', sort_columns=True).sort(_DATE_).fill_nan(0)
    qq = df.pivot(index=_DATE_, columns=_ASSET_, values=factor_quantile, aggregate_function='first', sort_columns=True).sort(_DATE_).fill_nan(-1)

    out = pd.DataFrame(index=rr[_DATE_])
    rr = rr.select(pl.exclude(_DATE_)).to_numpy() + 1  # 日收益
    qq = qq.select(pl.exclude(_DATE_)).to_numpy()  # 分组编号
    logger.info('多空收益准备数据,period={}', period)

    np.seterr(divide='ignore', invalid='ignore')

    # 等权
    w0 = qq == 0
    w9 = qq == q_max
    w0 = w0 / w0.sum(axis=1).reshape(-1, 1)
    w0 = np.where(w0 == w0, w0, 0)
    w9 = w9 / w9.sum(axis=1).reshape(-1, 1)
    w9 = np.where(w9 == w9, w9, 0)
    ww = (w9 - w0) / 2  # 除2，权重绝对值和一定要调整为1，否则后面会计算错误

    # 整行都为0，将其设成nan，后面计算时用于判断是否为0
    ww[(ww == 0).all(axis=1), :] = np.nan
    w0[(w0 == 0).all(axis=1), :] = np.nan
    w9[(w9 == 0).all(axis=1), :] = np.nan

    # 曲线的翻转
    out['1-G0,w=+1'] = 1 - cumulative_returns(rr, w0, funds=period, freq=period)
    # 权重的翻转。资金发生了变化。如果资金不共享，无法完全对冲
    out['G0-1,w=-1'] = cumulative_returns(rr, -w0, funds=period, freq=period) - 1

    out[f'G{q_max},w=+1'] = cumulative_returns(rr, w9, funds=period, freq=period)
    # 资金是共享的，每次调仓时需要将资金平分成两份
    out[f'G{q_max}~G0,w=+.5/-.5'] = cumulative_returns(rr, ww, funds=period, freq=period, init_cash=1.0)
    logger.info('多空收益计算完成,period={}\n{}', period, out.tail(1).to_string())
    return out


def calc_cum_return_weights(df: pl.DataFrame, fwd_ret_1: str, period: int = 1) -> pd.DataFrame:
    """指定权重计算收益。不再分层计算。资金也不分份"""
    rr = df.pivot(index=_DATE_, columns=_ASSET_, values=fwd_ret_1, aggregate_function='first', sort_columns=True).sort(_DATE_)
    ww = df.pivot(index=_DATE_, columns=_ASSET_, values=_WEIGHT_, aggregate_function='first', sort_columns=True).sort(_DATE_)

    out = pd.DataFrame(index=rr[_DATE_], columns=rr.columns[1:])
    rr = rr.select(pl.exclude(_DATE_)).to_numpy()  # 日收益
    ww = ww.select(pl.exclude(_DATE_)).to_numpy()  # 权重
    logger.info('权重收益准备数据,period={}', period)

    np.seterr(divide='ignore', invalid='ignore')

    rr = np.where(rr == rr, rr, 0.0)
    # 累计收益分资产，资金不共享
    # 由于是每天换仓，所以不存在空头计算不准的问题
    out[:] = np.cumprod(rr * ww + 1, axis=0)

    logger.info('权重收益计算完成,period={}\n{}', period, out.tail(1).to_string())
    return out


def create_portfolio2_sheet(df: pl.DataFrame,
                            fwd_ret_1: str,
                            *,
                            axvlines=()) -> None:
    """分资产收益。权重由外部指定，资金是隔离"""
    # 各资产收益，如果资产数量过多，图会比较卡顿
    df_cum_ret = calc_cum_return_weights(df, fwd_ret_1, 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), squeeze=False)
    axes = axes.flatten()
    # 分资产收益
    plot_quantile_portfolio(df_cum_ret, fwd_ret_1, axvlines=axvlines, ax=axes[0])

    # 资产平均收益，相当于等权
    s = df_cum_ret.mean(axis=1)
    s.name = 'portfolio'
    plot_quantile_portfolio(s, fwd_ret_1, axvlines=axvlines, ax=axes[1])
    fig.tight_layout()
