import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from alphainspect.utils import cumulative_returns


def calc_cum_return_by_quantile(df_pl: pl.DataFrame, fwd_ret_1: str, period: int = 5) -> pd.DataFrame:
    df_pd = df_pl.to_pandas().set_index(['date', 'asset'])
    rr = df_pd[fwd_ret_1].unstack()  # 1日收益率
    q_max = df_pd['factor_quantile'].max()
    pp = df_pd['factor_quantile'].unstack()  # 信号仓位

    out = pd.DataFrame(index=rr.index)
    rr = rr.to_numpy()
    pp = pp.to_numpy()
    for i in range(int(q_max) + 1):
        out[f'G{i}'] = cumulative_returns(rr, pp == i, period=period, is_mean=True)
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
                           groups=('G0', 'G9'),
                           axvlines=()) -> None:
    df_cum_ret = calc_cum_return_by_quantile(df_pl, fwd_ret_1, period)

    fix, axes = plt.subplots(2, 1, figsize=(12, 9))
    plot_quantile_portfolio(df_cum_ret, fwd_ret_1, period, axvlines=axvlines, ax=axes[0])
    for i, g in enumerate(groups):
        ax = plt.subplot(223 + i)
        plot_portfolio_heatmap(df_cum_ret, group=g, ax=ax)
