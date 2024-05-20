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

from alphainspect.events import with_around_price, create_events_sheet
from alphainspect.utils import with_factor_quantile

df_output = pl.read_parquet('data/data.parquet')

# %%
period = 5
axvlines = ('2020-01-01',)

factor = 'STD_010'  # 考察因子
fwd_ret_1 = 'RETURN_OO_05'  # 计算净值用的1日收益率
forward_return = 'RETURN_OO_05'  # 计算因子IC用的5日收益率

df_output = with_factor_quantile(df_output, factor, quantiles=10, factor_quantile='_fq_1')
df_output = with_around_price(df_output, 'CLOSE', periods_before=5, periods_after=15)

# %%
create_events_sheet(df_output, pl.col('STD_010') > 0.005, factor_quantile='_fq_1', axvlines=axvlines)

plt.show()

# %%
