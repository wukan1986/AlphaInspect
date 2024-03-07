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

from alphainspect.reports import create_2x2_sheet, create_3x2_sheet
from alphainspect.utils import with_factor_quantile

df_output = pl.read_parquet('data/data.parquet')

# %%
period = 5
axvlines = ('2020-01-01',)

factor = 'SMA_010'  # 考察因子
fwd_ret_1 = 'RETURN_OO_1'  # 计算净值用的1日收益率
forward_return = 'RETURN_OO_5'  # 计算因子IC用的5日收益率

df_output = with_factor_quantile(df_output, factor, quantiles=10)

# %%
create_2x2_sheet(df_output, factor, forward_return, fwd_ret_1, periods=(5, 10), axvlines=axvlines)
# %%
create_3x2_sheet(df_output, factor, forward_return, fwd_ret_1, period=period, axvlines=axvlines)

plt.show()

# %%
