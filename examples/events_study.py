"""
事件分析

1. 分析事件发生时，因子分层效果是否更好
2. 分析事件发生时，平均累计收益

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

from alphainspect.events import with_around_price, create_events_sheet
from alphainspect.utils import with_factor_quantile

df = pl.read_parquet('data/data.parquet')
# 生成前5后15价格
df = with_around_price(df, 'CLOSE', periods_before=5, periods_after=15)
# %%
axvlines = ('2020-01-01', '2024-01-01',)
factor = 'STD_010'  # 考察因子
fwd_ret_1 = 'RETURN_OO_05'  # 计算净值用的1日收益率

# %% 因子分层
df = with_factor_quantile(df, factor, quantiles=3, factor_quantile='_fq_1')

# %% 贝叶斯
# 检测哪种条件下，分层更明显
create_events_sheet(df, pl.col('STD_010') > 0.005, fwd_ret_1, factor_quantile='_fq_1', axvlines=axvlines)
create_events_sheet(df, pl.col('STD_010') <= 0.005, fwd_ret_1, factor_quantile='_fq_1', axvlines=axvlines)

# %%
plt.show()
