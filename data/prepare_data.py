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

# 生成多期收益率并移动，用于计算IC等信息
from codes.forward_returns import main

df = main(df)

# 计算因子
from codes.factors import main

df = main(df)

# save
df.write_parquet('data.parquet')
