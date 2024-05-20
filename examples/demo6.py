# %%
import os
import sys
from pathlib import Path

import pandas as pd

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
# ===============
import polars as pl

from alphainspect.reports import create_1x3_sheet, fig_to_img, html_template
from alphainspect.utils import with_factor_quantile


def func(factor, tpl: str = html_template):
    df_output = pl.read_parquet('data/data.parquet')

    # %%
    axvlines = ('2020-01-01',)

    fwd_ret_1 = 'RETURN_OO_01'  # 计算净值用的1日收益率
    forward_return = 'RETURN_OO_05'  # 计算因子IC用的5日收益率

    df_output = with_factor_quantile(df_output, factor, quantiles=10, factor_quantile='_fq_1')

    # %%
    fig, ic_dict, hist_dict, cum, avg, std = create_1x3_sheet(df_output, factor, forward_return, fwd_ret_1, factor_quantile='_fq_1', axvlines=axvlines)

    s1 = cum.iloc[-1]
    s2 = pd.Series(ic_dict | hist_dict)
    txt1 = pd.concat([s1, s2])
    txt1 = pd.DataFrame({factor: txt1}).T.to_html(float_format=lambda x: format(x, ',.4f'))  # , index=pd.Index([factor], name='factor')
    txt4 = fig_to_img(fig)

    tpl = tpl.replace('{{body}}', f'{txt1}\n{txt4}')

    with open(f'examples/{factor}.html', "w", encoding="utf-8") as f:
        f.write(tpl)

    return 0


if __name__ == '__main__':
    import multiprocessing

    # 没必要设置太大，因为部分计算使用的polars多线程，会将CPU跑满
    with multiprocessing.Pool(8) as pool:
        _map = map  # pool.map

        factors = ['SMA_005', ]
        output = list(_map(func, factors))
        print(output)
