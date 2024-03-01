# %%
import os
import sys
from pathlib import Path

from matplotlib import pyplot as plt

from alphainspect.ic import create_ic2_sheet

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
# ===============
# %%
import polars as pl

df_output = pl.read_parquet('data/data.parquet')
# %%
period = 5
axvlines = ('2020-01-01',)

factors = ['STD_010', 'STD_020', 'SMA_010', 'SMA_020']  # 考察因子
forward_returns = ['RETURN_CC_1', 'RETURN_OO_1', 'RETURN_OO_2', 'RETURN_OO_5']  # 同一因子，不同持有期对比

create_ic2_sheet(df_output, factors, forward_returns)
plt.show()
