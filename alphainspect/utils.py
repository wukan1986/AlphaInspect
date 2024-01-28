import numpy as np
import pandas as pd
import polars as pl
from polars import Series, Expr, Int16

from alphainspect import _QUANTILE_, _DATE_, _GROUP_
from alphainspect._nb import _sub_portfolio_returns


def rank_ic(a: str, b: str) -> Expr:
    """RankIC"""
    return pl.corr(a, b, method='spearman', ddof=0, propagate_nans=False)


def auto_corr(a: str, period: int) -> Expr:
    """自相关"""
    return pl.corr(pl.col(a), pl.col(a).shift(period), method='spearman', ddof=0, propagate_nans=False)


def _qcut(x: Series, q: int) -> Series:
    # TODO 等待提供
    if x.null_count() == len(x):
        return x
    else:
        return pd.qcut(x, q, labels=False, duplicates='drop')


def cs_bucket(x: Expr, q: int = 10) -> Expr:
    """Convert float values into indexes for user-specified buckets. Bucket is useful for creating group values, which can be passed to group operators as input."""
    # TODO 等官方提供原生功能
    return x.map_batches(lambda x1: Series(_qcut(x1, q), nan_to_null=True, dtype=Int16))


def with_factor_quantile(df_pl: pl.DataFrame, factor: str, quantiles: int = 10, by_group: bool = False) -> pl.DataFrame:
    """添加因子分位数信息

    Parameters
    ----------
    df_pl
    factor: str
        因子名
    quantiles: int
        分层数
    by_group:bool
        是否分组

    Returns
    -------
    pl.DataFrame

    """

    def _func_cs(df: pl.DataFrame):
        return df.with_columns([
            cs_bucket(pl.col(factor), quantiles).alias(_QUANTILE_),
        ])

    # 将nan改成null
    df_pl = df_pl.with_columns(pl.col(factor).fill_nan(None))

    if by_group:
        return df_pl.group_by(by=[_DATE_, _GROUP_]).map_groups(_func_cs)
    else:
        return df_pl.group_by(by=[_DATE_]).map_groups(_func_cs)


def cumulative_returns(returns: np.ndarray, weights: np.ndarray,
                       funds: int = 3, freq: int = 3,
                       benchmark: np.ndarray = None,
                       ret_mean: bool = True,
                       init_cash: float = 1.0) -> np.ndarray:
    """累积收益

    精确计算收益是非常麻烦的事情，比如考虑手续费、滑点、涨跌停无法入场。考虑过多也会导致计算量巨大。
    这里只做估算，用于不同因子之间收益比较基本够用。更精确的计算请使用专用的回测引擎

    需求：因子每天更新，但策略是持仓3天
    1. 每3天取一次因子，并持有3天。即入场时间对净值影响很大。净值波动剧烈
    2. 资金分成3份，每天入场一份。每份隔3天做一次调仓，多份资金不共享。净值波动平滑

    本函数使用的第2种方法，例如：某支股票持仓信息如下
    [0,1,1,1,0,0]
    资金分成三份，每次持有三天，
    [0,0,0,1,1,1] # 第0、3、6...位，fill后两格
    [0,1,1,1,0,0] # 第1、4、7...位，fill后两格
    [0,0,1,1,1,0] # 第2、5、8...位，fill后两格

    Parameters
    ----------
    returns: np.ndarray
        1期简单收益率。自动记在出场位置。
    weights: np.ndarray
        持仓权重。需要将信号移动到出场日期。权重绝对值和
    funds: int
        资金拆成多少份
    freq:int
        再调仓频率
    benchmark: 1d np.ndarray
        基准收益率
    ret_mean: bool
        返回多份资金合成曲线
    init_cash: float
        初始资金

    Returns
    -------
    np.ndarray

    References
    ----------
    https://github.com/quantopian/alphalens/issues/187

    """
    # 一维修改成二维，代码统一
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)

    # 记录有效数
    vailds = np.isfinite(weights).any(axis=1)
    # 修正数据中出现的nan
    returns = np.where(returns == returns, returns, 1.0)
    # 权重需要已经分配好，绝对值和为1
    weights = np.where(weights == weights, weights, 0.0)

    # 形状
    m, n = weights.shape

    #  记录每份资金每期收益率
    out = _sub_portfolio_returns(m, n, weights, returns, vailds, funds, freq, init_cash)
    if ret_mean:
        if benchmark is None:
            # 多份净值直接叠加后平均
            return out.mean(axis=1)
        else:
            # 有基准，计算超额收益
            return out.mean(axis=1) - (benchmark + 1).cumprod()
    else:
        return out
