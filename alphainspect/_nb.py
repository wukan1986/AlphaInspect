import numpy as np
from numba import jit, prange


@jit(nopython=True, nogil=True, fastmath=True)
def np_apply_along_axis_1d(func1d, axis, arr, out):
    if axis == 0:
        for i in range(len(out)):
            out[i] = func1d(arr[:, i])
    else:
        for i in range(len(out)):
            out[i] = func1d(arr[i, :])
    return out


@jit(nopython=True, nogil=True, fastmath=True)
def np_apply_along_axis_2d(func1d, axis, arr, out):
    if axis == 0:
        for i in range(out.shape[1]):
            out[:, i] = func1d(arr[:, i])
    else:
        for i in range(out.shape[0]):
            out[i, :] = func1d(arr[i, :])
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def np_mean(arr, axis, out):
    return np_apply_along_axis_1d(np.mean, axis, arr, out)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def np_sum(arr, axis, out):
    # sum支持axis
    return np_apply_along_axis_1d(np.sum, axis, arr, out)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def np_cumprod(arr, axis, out):
    return np_apply_along_axis_2d(np.cumprod, axis, arr, out)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def np_tile(arr, reps):
    m, n = arr.shape
    out = np.empty(shape=(m, n * reps), dtype=arr.dtype)
    for i in range(reps):
        out[:, i * n:(i + 1) * n] = arr
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True, parallel=True)
def _sub_portfolio_returns_x1(m: int, n: int, weights: np.ndarray, returns: np.ndarray, period: int = 3) -> np.ndarray:
    """一份资金每天的平均收益率

    但由于第二天起始权重其实不同，所以收益率等权已经不一样了，有误差，但速度快
    """
    # tile时可能长度不够，所以补充一段
    weights = np.concatenate((weights, weights[:period]), axis=0)
    # 记录每份的收益率
    out = np.zeros(shape=(m, period), dtype=float)
    # 资金分成period份
    for i in prange(period):
        # 某一天的持仓需要持续period天
        w = np_tile(weights[i::period], period).reshape(-1, n)
        if i > 0:
            # shift操作
            w[i:] = w[:-i]
            w[:i] = 0
        w = w[:m]

        # 计算此份资金的收益，净值从1开始
        np_sum(returns * w, axis=1, out=out[:, i])

    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True, parallel=True)
def _sub_portfolio_returns(m: int, n: int, weights: np.ndarray, returns: np.ndarray, period: int = 3, start=0) -> np.ndarray:
    """资金分成N份。每隔N天取一次权重，这N天内每次都用上次的市值乘以权重得到新市值，乘以收益率得到日终市值"""
    # 记录每份的收益率
    out = np.zeros(shape=(m, period), dtype=float)
    # 资金分成period份
    for i in prange(period):
        val = np.zeros(shape=(m, n), dtype=float)
        val[:, 0] = 1
        for j in range(start + i, m):
            if j % period == i:
                # 取一行权重。!!!之后权重不能全0。因为清仓后资金没有维护
                val[j] = np.sum(val[j - 1]) * weights[j]
            else:
                # 取昨天的市值
                val[j] = val[j - 1]
            val[j] = returns[j] * val[j]
        np_sum(val, axis=1, out=out[:, i])

    return out
