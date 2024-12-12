# %%
import os
import sys
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
# ===============
import multiprocessing
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from alphainspect.reports import create_1x3_sheet, fig_to_img, html_template
from alphainspect.utils import with_factor_quantile

INPUT1_PATH = r'data/data.parquet'
OUTPUT_PATH = r'output'
output = Path(OUTPUT_PATH)
output.mkdir(parents=True, exist_ok=True)


def func(kv):
    name, factors = kv
    axvlines = ('2020-01-01', '2024-01-01',)
    fwd_ret_1 = 'RETURN_OO_05'  # 计算净值用的1日收益率
    quantiles = 5

    tbl = {}
    df_mean = {}
    df_std = {}
    df_last = {}
    imgs = []

    df = pl.read_parquet(INPUT1_PATH)
    for factor in factors:
        df = with_factor_quantile(df, factor, quantiles=quantiles, factor_quantile=f'_fq_{factor}')

    for factor in factors:
        fig, ic_dict, hist_dict, cum, avg, std = create_1x3_sheet(df, factor, fwd_ret_1, factor_quantile=f'_fq_{factor}', axvlines=axvlines)

        s1 = cum.iloc[-1]
        df_last[factor] = s1
        df_mean[factor] = avg
        df_std[factor] = std

        s2 = {'monotonic': np.sign(s1.diff()).sum()}
        s3 = pd.Series(s2 | ic_dict | hist_dict)
        tbl[factor] = pd.concat([s1, s3])
        imgs.append(fig_to_img(fig))

    df_last = pd.DataFrame(df_last)
    df_mean = pd.DataFrame(df_mean)
    df_std = pd.DataFrame(df_std)

    # 各指标柱状图
    tbl = pd.DataFrame(tbl)
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    ax = df_last.plot.bar(ax=axes[0])
    ax.set_title(f'Last Total Return By Quantile')
    ax = df_mean.plot.bar(ax=axes[1])
    ax.set_title(f'Mean Return By Quantile')
    ax = df_std.plot.bar(ax=axes[2])
    ax.set_title(f'Std Return By Quantile')
    plt.xticks(rotation=0)
    fig.tight_layout()
    imgs.insert(0, fig_to_img(fig))

    # 表格
    txt1 = tbl.T.to_html(float_format=lambda x: format(x, '.4f'))
    # 图
    txt2 = '\n'.join(imgs)
    tpl = html_template.replace('{{body}}', f'{txt1}\n{txt2}')

    with open(str(output / f'{name}.html'), "w", encoding="utf-8") as f:
        f.write(tpl)

    return 0


if __name__ == '__main__':
    factors_kv = {
        "SMA": ['SMA_005', 'SMA_010', 'SMA_020', ],
        "STD": ['STD_005', 'STD_010', 'STD_020', ],
    }
    t0 = time.perf_counter()
    logger.info('开始')
    # 没必要设置太大，因为部分计算使用的polars多线程，会将CPU跑满
    with multiprocessing.Pool(2) as pool:
        print(list(pool.map(func, factors_kv.items())))
    logger.info('结束')
    logger.info(f'耗时：{time.perf_counter() - t0:.2f}s')
    os.system(f'explorer.exe "{output}"')
