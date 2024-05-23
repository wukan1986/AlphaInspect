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
from alphainspect.returns import create_returns2_sheet

df_output = pl.read_parquet('data/data.parquet')
# %%
factor1 = 'SMA_010'  # 考察因子
factor2 = 'STD_010'  # 考察因子
forward_return = "RETURN_OO_01"  # 同一因子，不同持有期对比

# %%
# 独立双重排序
df_output = with_factor_quantile(df_output, factor1, quantiles=3, factor_quantile='_fq_1')
df_output = with_factor_quantile(df_output, factor2, quantiles=5, factor_quantile='_fq_2')
create_returns2_sheet(df_output, forward_return, ['_fq_1', '_fq_2'])

# 条件双重排序
df_output = with_factor_quantile(df_output, factor1, quantiles=3, factor_quantile='_fq_1')
df_output = with_factor_quantile(df_output, factor2, quantiles=5, factor_quantile='_fq_2', group_name='_fq_1')
create_returns2_sheet(df_output, forward_return, ['_fq_1', '_fq_2'])

plt.show()

# %%
