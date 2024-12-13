"""
剔除高相关性因子

防止多重共线性
"""
# %%
import os
import sys
from pathlib import Path
from pprint import pprint

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
# ===============
# %%
import polars as pl
from matplotlib import pyplot as plt

from alphainspect.ic import create_ic2_sheet
from alphainspect.selection import drop_above_corr_thresh
from alphainspect.utils import select_by_suffix

df = pl.read_parquet('data/data.parquet')
# %%
factors = ['STD_010', 'STD_020', 'STD_060', 'SMA_010', 'SMA_020', 'SMA_060']  # 考察因子
forward_returns = ['RETURN_CC_01', 'RETURN_OO_01', 'RETURN_OO_02', 'RETURN_OO_05', 'RETURN_OO_10']  # 同一因子，不同持有期对比
# 多图IC对比
df_ic = create_ic2_sheet(df, factors, forward_returns)

# 选出指定后缀的列
df_pa = select_by_suffix(df_ic, '__RETURN_OO_05')
cols_to_drop, above_thresh_pairs = drop_above_corr_thresh(df_pa, thresh=0.6)
# 需要剔除的因子
print(cols_to_drop)
pprint(above_thresh_pairs)

plt.show()
