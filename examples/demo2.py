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

from alphainspect.ic import create_ic_sheet
from alphainspect.portfolio import create_portfolio_sheet
from alphainspect.returns import create_returns_sheet
from alphainspect.turnover import create_turnover_sheet
from alphainspect.utils import with_factor_quantile

df_output = pl.read_parquet('data/data.parquet')
# %%
period = 5
axvlines = ('2020-01-01',)

factor = 'STD_010'  # 考察因子
forward_returns = ['RETURN_CC_1', 'RETURN_OO_1', 'RETURN_OO_2', 'RETURN_OO_5']  # 同一因子，不同持有期对比

# %%
df_output = with_factor_quantile(df_output, factor, quantiles=10)
# %%
# IC统计
create_ic_sheet(df_output, factor, forward_returns, method='rank_ic')
# %%
# 收益率统计
create_returns_sheet(df_output, factor, forward_returns)
# %%
fwd_ret_1 = 'RETURN_OO_1'  # 计算净值必需提供1日收益率
create_portfolio_sheet(df_output, fwd_ret_1, period=5, axvlines=axvlines)
create_turnover_sheet(df_output, factor, periods=(1, 5, 10, 20), axvlines=axvlines)

plt.show()

# %%
