import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from alphainspect import _QUANTILE_, _DATE_, _ASSET_
from alphainspect.utils import cumulative_returns


def calc_cum_return_by_quantile(df_pl: pl.DataFrame, fwd_ret_1: str, period: int = 5) -> pd.DataFrame:
    df_pd = df_pl.to_pandas().set_index([_DATE_, _ASSET_])
    q_max = int(df_pd[_QUANTILE_].max())
    rr = df_pd[fwd_ret_1].unstack().fillna(0)  # 1日收益率
    qq = df_pd[_QUANTILE_].unstack().fillna(-1)  # 因子所在分组编号

    out = pd.DataFrame(index=rr.index)
    rr = rr.to_numpy() + 1  # 日收益
    qq = qq.to_numpy()  # 分组编号
    np.seterr(divide='ignore', invalid='ignore')
    for i in range(int(q_max) + 1):
        # 等权
        b = qq == i
        d = b / b.sum(axis=1).reshape(-1, 1)
        d[(d == 0).all(axis=1), :] = np.nan
        # 权重绝对值和为1
        out[f'G{i}'] = cumulative_returns(rr, d, funds=period, freq=period)
    # !!!直接减是错误的，因为两资金是独立的，资金减少的一份由于资金不足对冲比例已经不再是1:1
    # out['spread'] = out[f'G{q_max}'] - out[f'G0']
    return out


def calc_cum_return_spread(df_pl: pl.DataFrame, fwd_ret_1: str, period: int = 5) -> pd.DataFrame:
    df_pd = df_pl.to_pandas().set_index([_DATE_, _ASSET_])
    q_max = int(df_pd[_QUANTILE_].max())
    rr = df_pd[fwd_ret_1].unstack().fillna(0)  # 1日收益率
    qq = df_pd[_QUANTILE_].unstack().fillna(-1)  # 因子所在分组编号

    out = pd.DataFrame(index=rr.index)
    rr = rr.to_numpy() + 1  # 日收益
    qq = qq.to_numpy()  # 分组编号
    np.seterr(divide='ignore', invalid='ignore')

    # 等权
    b0 = qq == 0
    b9 = qq == q_max
    b0 = b0 / b0.sum(axis=1).reshape(-1, 1)
    b0 = np.where(b0 == b0, b0, 0)
    b9 = b9 / b9.sum(axis=1).reshape(-1, 1)
    b9 = np.where(b9 == b9, b9, 0)
    bb = (b9 - b0) / 2  # 除2，权重绝对值和一定要调整为1，否则后面会计算错误

    # 整行都为0，将其设成nan，后面计算时用于判断是否为0
    bb[(bb == 0).all(axis=1), :] = np.nan
    b0[(b0 == 0).all(axis=1), :] = np.nan
    b9[(b9 == 0).all(axis=1), :] = np.nan

    # 曲线的翻转
    out['1-G0 w=+1'] = 1 - cumulative_returns(rr, b0, funds=period, freq=period)
    # 权重的翻转。资金发生了变化。如果资金不共享，无法完全对冲
    out['G0-1 w=-1'] = cumulative_returns(rr, -b0, funds=period, freq=period) - 1

    out[f'G{q_max} w=+1'] = cumulative_returns(rr, b9, funds=period, freq=period)
    # 资金是共享的，每次调仓时需要将资金平分成两份
    out[f'G{q_max}~G0 w=+.5/-.5'] = cumulative_returns(rr, bb, funds=period, freq=period, init_cash=1.0)

    return out


def plot_quantile_portfolio(df_pd: pd.DataFrame, fwd_ret_1: str, period: int = 5,
                            *,
                            axvlines=None, ax=None) -> None:
    ax = df_pd.plot(ax=ax, title=f'{fwd_ret_1}, period={period}', cmap='coolwarm', lw=1, grid=True)
    ax.legend(loc='upper left')
    ax.set_xlabel('')
    for v in axvlines:
        ax.axvline(x=v, c="b", ls="--", lw=1)


def plot_portfolio_heatmap(df_pd: pd.DataFrame,
                           *,
                           group='G9', ax=None) -> None:
    """月度热力图。可用于IC, 收益率等"""
    out = pd.DataFrame(index=df_pd.index)
    out['year'] = out.index.year
    out['month'] = out.index.month
    out['first'] = df_pd[group]
    out['last'] = df_pd[group]
    out = out.groupby(by=['year', 'month']).agg({'first': 'first', 'last': 'last'})
    out['cum_ret'] = out['last'] / out['first'] - 1
    ax = sns.heatmap(out['cum_ret'].unstack(), annot=True, cmap='RdYlGn_r', cbar=False, annot_kws={"size": 7}, ax=ax)
    ax.set_title(f"{group},Monthly Return")
    ax.set_xlabel('')


def create_portfolio_sheet(df_pl: pl.DataFrame,
                           fwd_ret_1: str,
                           period=5,
                           *,
                           axvlines=()) -> None:
    df_cum_ret = calc_cum_return_by_quantile(df_pl, fwd_ret_1, period)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    plot_quantile_portfolio(df_cum_ret, fwd_ret_1, period, axvlines=axvlines, ax=axes[0])
    groups = df_cum_ret.columns[[0, -1]]
    for i, g in enumerate(groups):
        ax = plt.subplot(223 + i)
        plot_portfolio_heatmap(df_cum_ret, group=g, ax=ax)
    fig.tight_layout()

    df_spread = calc_cum_return_spread(df_pl, fwd_ret_1, period)
    fig, axes = plt.subplots(1, 1, figsize=(12, 9))
    plot_quantile_portfolio(df_spread, fwd_ret_1, period, axvlines=axvlines, ax=axes)
    fig.tight_layout()
