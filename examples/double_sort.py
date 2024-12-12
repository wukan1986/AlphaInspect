"""
双重排序
1. 独立。对因子1进行分位数分组，对因子2进行分位数分组。分组后，每组数量差距很大
2. 条件。对因子1进行分位数分组，组内对因子2进行分位数分组合，分组后，每组数量差距不大

"""
# %%
import os
import sys
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
# ===============
# %%
import matplotlib.pyplot as plt
import polars as pl

from alphainspect.utils import with_factor_quantile
from alphainspect.plotting import create_describe2_sheet

df = pl.read_parquet('data/data.parquet')
# %%
factor1 = 'SMA_010'  # 考察因子
factor2 = 'STD_010'  # 考察因子
fwd_ret_1 = "RETURN_OO_05"  # 因子对标的收益率

# %%
# 独立双重排序
df = with_factor_quantile(df, factor1, quantiles=3, factor_quantile='_fq_1')
df = with_factor_quantile(df, factor2, quantiles=5, factor_quantile='_fq_2')
create_describe2_sheet(df, fwd_ret_1, ['_fq_1', '_fq_2'])

# 条件双重排序
df = with_factor_quantile(df, factor1, quantiles=3, factor_quantile='_fq_1')
df = with_factor_quantile(df, factor2, quantiles=5, factor_quantile='_fq_2', group_name='_fq_1')
create_describe2_sheet(df, fwd_ret_1, ['_fq_1', '_fq_2'])

plt.show()

# %%
