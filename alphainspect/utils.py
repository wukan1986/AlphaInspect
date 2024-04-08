import math
from typing import Dict

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from loguru import logger
from polars import selectors as cs

from alphainspect import _QUANTILE_, _DATE_, _GROUP_
from alphainspect._nb import _sub_portfolio_returns


def rank_qcut(x: pl.Expr, q: int = 10) -> pl.Expr:
    """结果与qcut基本一样，速度快三倍"""
    # TODO 等官方提供原生功能
    a = x.rank(method='min') - 1.001
    b = pl.max_horizontal(x.count() - 1, 1)
    return (a / b * q).cast(pl.Int16)


def with_factor_quantile(df_pl: pl.DataFrame, factor: str, quantiles: int = 10, by_group: bool = False, factor_quantile: str = _QUANTILE_) -> pl.DataFrame:
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
    factor_quantile:str
        分组名

    Returns
    -------
    pl.DataFrame

    """

    def _func_cs(df: pl.DataFrame):
        return df.with_columns([
            rank_qcut(pl.col(factor), quantiles).alias(factor_quantile),
        ])

    # 将nan改成null
    df_pl = df_pl.with_columns(pl.col(factor).fill_nan(None))

    if by_group:
        return df_pl.group_by(by=[_DATE_, _GROUP_]).map_groups(_func_cs)
    else:
        return df_pl.group_by(_DATE_).map_groups(_func_cs)


def with_quantile_tradable(df_pl: pl.DataFrame, factor_quantile: str, next_doji: str = 'NEXT_DOJI') -> pl.DataFrame:
    """是否可以交易，将不可产易的分到其它分组

    Parameters
    ----------
    df_pl: pl.DataFrame
    factor_quantile: str
        分组名
    next_doji: str
        明日涨跌停。修改factor_quantile到-1组

    Returns
    -------
    pl.DataFrame

    """
    if next_doji is not None:
        pl.when(pl.col(next_doji)).then(-1).otherwise(pl.col(factor_quantile)).name.keep(),
    return df_pl


def cumulative_returns(returns: np.ndarray, weights: np.ndarray,
                       funds: int = 3, freq: int = 3,
                       benchmark: np.ndarray = None,
                       ret_mean: bool = True,
                       init_cash: float = 1.0,
                       risk_free: float = 1.0,  # 1.0 + 0.025 / 250
                       ) -> np.ndarray:
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
    risk_free: float
        无风险收益率。用在现金列。空仓时，可以给现金提供利息

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

    # 形状
    m, n = weights.shape

    # 现金权重
    weights_cash = 1 - np.round(np.nansum(np.abs(weights), axis=1), 5)
    # TODO 也可以添加两列现金，一列有利息，一列没利息。细节要按策略进行定制
    returns = np.concatenate((np.ones(shape=(m, 1), dtype=returns.dtype), returns), axis=1)
    weights = np.concatenate((np.zeros(shape=(m, 1), dtype=weights.dtype), weights), axis=1)
    # 添加第0列做为现金，用于处理CTA空仓的问题
    weights[:, 0] = weights_cash
    # 可以考虑给现金指定一个固定收益
    returns[:, 0] = risk_free

    # 修正数据中出现的nan
    returns = np.where(returns == returns, returns, 1.0)
    # 权重需要已经分配好，绝对值和为1
    weights = np.where(weights == weights, weights, 0.0)

    # 新形状
    m, n = weights.shape

    #  记录每份资金每期收益率
    out = _sub_portfolio_returns(m, n, weights, returns, funds, freq, init_cash)
    if ret_mean:
        if benchmark is None:
            # 多份净值直接叠加后平均
            return out.mean(axis=1)
        else:
            # 有基准，计算超额收益
            return out.mean(axis=1) - (benchmark + 1).cumprod()
    else:
        return out


def plot_heatmap(df_pd: pd.DataFrame,
                 *,
                 title='Mean IC',
                 ax=None) -> None:
    """多个IC的热力图"""
    # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    ax = sns.heatmap(df_pd, annot=True, cmap='RdYlGn_r', cbar=False, annot_kws={"size": 7}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')


def get_row_col(count: int):
    """通过图总数，得到二维数量"""
    len_sqrt = math.sqrt(count)
    row, col = math.ceil(len_sqrt), math.floor(len_sqrt)
    if row * col < count:
        col += 1
    return row, col


def select_by_suffix(df_pl: pl.DataFrame, name: str) -> pl.DataFrame:
    """选择指定后缀的所有因子"""
    return df_pl.select(cs.ends_with(name).name.map(lambda x: x[:-len(name)]))


def select_by_prefix(df_pl: pl.DataFrame, name: str) -> pl.DataFrame:
    """选择指定前缀的所有因子"""
    return df_pl.select(cs.starts_with(name).name.map(lambda x: x[len(name):]))


def plot_hist(df_pl: pl.DataFrame, col: str,
              *,
              kde: bool = False,  # 启用kde后速度慢了非常多
              ax=None) -> Dict[str, float]:
    """直方图

    Examples
    --------
    >>> plot_hist(df_pl, 'RETURN_OO_1')
    """
    a = df_pl[col].to_pandas().replace([-np.inf, np.inf], np.nan).dropna()

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
    title = f"{col},mean={mean:0.4f},std={std:0.4f},skew={skew:0.4f},kurt={kurt:0.4f}"
    logger.info(title)
    ax.set_title(title)
    ax.set_xlabel('')

    return {'mean': mean, 'std': std, 'skew': skew, 'kurt': kurt}


# =================================
# 没分好类的函数先放这，等以后再移动
def symmetric_orthogonal(matrix):
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # 按照特征值的大小排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 正交化矩阵
    orthogonal_matrix = np.linalg.qr(sorted_eigenvectors)[0]

    return orthogonal_matrix
