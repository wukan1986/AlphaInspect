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
from alphainspect.returns import create_returns_sheet
from alphainspect.turnover import create_turnover_sheet
from alphainspect.utils import with_factor_quantile

df_output = pl.read_parquet('data/data.parquet')
# %%
period = 5
axvlines = ('2020-01-01',)

factor = 'STD_010'  # 考察因子
forward_returns = ['RETURN_CC_01', 'RETURN_OO_01', 'RETURN_OO_02', 'RETURN_OO_05']  # 同一因子，不同持有期对比

# %%

# %%
# IC统计
create_ic1_sheet(df_output, factor, forward_returns, method="rank_ic")

df_output = with_factor_quantile(df_output, factor, quantiles=10, factor_quantile='_fq_1')
# %%
# 收益率统计
create_returns_sheet(df_output, factor, forward_returns, factor_quantile='_fq_1')
# %%
fwd_ret_1 = 'RETURN_OO_05'  # 计算净值必需提供1日收益率
create_portfolio1_sheet(df_output, fwd_ret_1, factor_quantile='_fq_1', axvlines=axvlines)
create_turnover_sheet(df_output, factor, periods=(1, 5, 10, 20), factor_quantile='_fq_1', axvlines=axvlines)

plt.show()

# %%
