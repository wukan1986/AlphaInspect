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

from alphainspect.reports import create_2x2_sheet, create_3x2_sheet, create_1x3_sheet
from alphainspect.utils import with_factor_quantile, with_factor_top_k  # noqa

df = pl.read_parquet('data/data.parquet')

# %%
axvlines = ('2020-01-01', '2024-01-01',)

factor = 'SMA_010'  # 考察因子
fwd_ret_1 = 'RETURN_OO_05'  # 计算净值用的1日收益率

df = with_factor_quantile(df, factor, quantiles=10, factor_quantile='_fq_1')
# df = with_factor_top_k(df, factor, top_k=20, factor_quantile='_fq_1')

# %%
fig, ic_dict, hist_dict, cum, avg, std = create_1x3_sheet(df, factor, fwd_ret_1, factor_quantile='_fq_1', axvlines=axvlines)
# %%
create_2x2_sheet(df, factor, fwd_ret_1, factor_quantile='_fq_1', axvlines=axvlines)
# %%
create_3x2_sheet(df, factor, fwd_ret_1, factor_quantile='_fq_1', axvlines=axvlines)

plt.show()

# %%
