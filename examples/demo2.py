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

from alphainspect.ic import create_ic1_sheet
from alphainspect.portfolio import create_portfolio1_sheet
from alphainspect.plotting import create_describe1_sheet
from alphainspect.turnover import create_turnover_sheet
from alphainspect.utils import with_factor_quantile

df = pl.read_parquet('data/data.parquet')
# %%
axvlines = ('2020-01-01', '2024-01-01',)

factor = 'STD_010'  # 考察因子
forward_returns = ['RETURN_CC_01', 'RETURN_OO_01', 'RETURN_OO_02', 'RETURN_OO_05']  # 同一因子，不同持有期对比
# forward_returns = ['RETURN_CC_01']
# forward_returns = ['STD_010']
# %%

# %%
# IC统计
create_ic1_sheet(df, factor, forward_returns, method="rank_ic")

df = with_factor_quantile(df, factor, quantiles=10, factor_quantile='_fq_1')
# %%
# 收益率统计
# create_describe_sheet(df, forward_returns, factor_quantile='_fq_1')
# %%
fwd_ret_1 = 'RETURN_OO_05'  # 计算净值必需提供1日收益率
# create_portfolio1_sheet(df, fwd_ret_1, factor_quantile='_fq_1', axvlines=axvlines)
# create_turnover_sheet(df, factor, periods=(1, 5, 10, 20), factor_quantile='_fq_1', axvlines=axvlines)

plt.show()

# %%
