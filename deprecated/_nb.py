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
def _sub_portfolio_returns(m: int, n: int,
                           weights: np.ndarray, returns: np.ndarray,
                           funds: int = 1,
                           freq: int = -1,
                           init_cash: float = 1.0) -> np.ndarray:
    """资金分成N份。每隔N天取一次权重，这N天内每次都用上次的市值乘以权重得到新市值，乘以收益率得到日终市值

    输出的数据全不为nan
    """
    if freq == -1:
        freq = m
    # 记录每份的收益率
    out = np.zeros(shape=(m, funds), dtype=float)
    # 资金分成funds份
    for i in prange(funds):
        cashflow = np.zeros(shape=(m, n), dtype=float)  # 多头为0，空头为2*weights
        val = np.zeros(shape=(m, n), dtype=float)
        val[:, 0] = init_cash  # 初始资金全放第0列
        last_sum = init_cash
        for j in range(i, m):
            # 调仓节点，如果设成m表示只第一天进行调仓
            if (j % freq == i) and (last_sum > 0):
                val[j] = last_sum * weights[j]
                # 负数变正数*2，正数变成0
                cashflow[j] = np.maximum(-val[j], 0) * 2
            else:
                # 不调仓，直接取昨天的现金和市值做为盘前值
                cashflow[j] = cashflow[j - 1]
                val[j] = val[j - 1]

            val[j] = returns[j] * val[j]
            last = val[j] + cashflow[j]  # 在这里，很有可能做空亏成负数
            last_sum = np.sum(last)
            if last_sum <= 0:
                # 亏损了，将市值移动到现金流中，不玩了
                cashflow[j] += val[j]
                val[j] = 0

        # 如果不水平求和，就能看到每支股票的累计收益
        np_sum(val + cashflow, axis=1, out=out[:, i])

    return out
