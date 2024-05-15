"""
准备测试数据

注意：本测试数据生成需要安装polars_ta，如果使用其它数据源可不安装此包
pip install polars_ta

"""
import numpy as np
import pandas as pd
import polars as pl

_N = 250 * 10
_K = 500

asset = [f's_{i:04d}' for i in range(_K)]
date = pd.date_range('2015-1-1', periods=_N)

df = pd.DataFrame({
    'OPEN': np.cumprod(1 + np.random.uniform(-0.01, 0.01, size=(_N, _K)), axis=0).reshape(-1),
    'HIGH': np.cumprod(1 + np.random.uniform(-0.01, 0.01, size=(_N, _K)), axis=0).reshape(-1),
    'LOW': np.cumprod(1 + np.random.uniform(-0.01, 0.01, size=(_N, _K)), axis=0).reshape(-1),
    'CLOSE': np.cumprod(1 + np.random.uniform(-0.01, 0.01, size=(_N, _K)), axis=0).reshape(-1),
    "FILTER": np.tri(_N, _K, k=-2).reshape(-1),
}, index=pd.MultiIndex.from_product([date, asset], names=['date', 'asset'])).reset_index()

# 向脚本输入数据
df = pl.from_pandas(df)
# 数据长度不同
df = df.filter(pl.col('FILTER') == 1)

"""
RETURN_OC_1 = ts_delay(CLOSE, -1) / ts_delay(OPEN, -1) - 1
RETURN_CC_1 = ts_delay(CLOSE, -1) / CLOSE - 1
RETURN_CO_1 = ts_delay(OPEN, -1) / CLOSE - 1
RETURN_OO_1 = ts_delay(OPEN, -2) / ts_delay(OPEN, -1) - 1
RETURN_OO_2 = (ts_delay(OPEN, -3) / ts_delay(OPEN, -1)) ** (1 / 2) - 1
RETURN_OO_5 = (ts_delay(OPEN, -6) / ts_delay(OPEN, -1)) ** (1 / 5) - 1
RETURN_OO_10 = (ts_delay(OPEN, -11) / ts_delay(OPEN, -1)) ** (1 / 10) - 1

"""
# 生成多期收益率并移动，用于计算IC等信息
from codes.forward_returns import main

df = main(df)
"""
HHV_005 = ts_max(HIGH, 5) / CLOSE
HHV_010 = ts_max(HIGH, 10) / CLOSE
HHV_020 = ts_max(HIGH, 20) / CLOSE
HHV_060 = ts_max(HIGH, 60) / CLOSE
HHV_120 = ts_max(HIGH, 120) / CLOSE

LLV_005 = ts_min(LOW, 5) / CLOSE
LLV_010 = ts_min(LOW, 10) / CLOSE
LLV_020 = ts_min(LOW, 20) / CLOSE
LLV_060 = ts_min(LOW, 60) / CLOSE
LLV_120 = ts_min(LOW, 120) / CLOSE

SMA_005 = ts_mean(CLOSE, 5) / CLOSE
SMA_010 = ts_mean(CLOSE, 10) / CLOSE
SMA_020 = ts_mean(CLOSE, 20) / CLOSE
SMA_060 = ts_mean(CLOSE, 60) / CLOSE
SMA_120 = ts_mean(CLOSE, 120) / CLOSE

STD_005 = ts_std_dev(CLOSE, 5) / CLOSE
STD_010 = ts_std_dev(CLOSE, 10) / CLOSE
STD_020 = ts_std_dev(CLOSE, 20) / CLOSE
STD_060 = ts_std_dev(CLOSE, 60) / CLOSE
STD_120 = ts_std_dev(CLOSE, 120) / CLOSE

ROCP_001 = ts_returns(CLOSE, 1)
ROCP_003 = ts_returns(CLOSE, 3)
ROCP_005 = ts_returns(CLOSE, 5)
ROCP_010 = ts_returns(CLOSE, 10)
ROCP_020 = ts_returns(CLOSE, 20)
ROCP_060 = ts_returns(CLOSE, 60)
ROCP_120 = ts_returns(CLOSE, 120)
"""
# 计算因子
from codes.factors import main

df = main(df)

# save
df.write_parquet('data.parquet', compression='zstd')
