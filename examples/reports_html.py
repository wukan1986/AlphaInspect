"""
HTML报告

省去了Notebook的转换问题，速度更快一点点，同时可以将统计结果放在最开头
支持复杂的参数传递

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
import multiprocessing
import time

import polars as pl
from loguru import logger

from alphainspect.reports import report_html
from alphainspect.utils import with_factor_quantile, with_factor_top_k  # noqa

INPUT1_PATH = r'data/data.parquet'
OUTPUT_PATH = r'output'


def func(kv):
    name, factors = kv

    df = pl.read_parquet(INPUT1_PATH)

    report_html(name, factors, df, OUTPUT_PATH,
                fwd_ret_1='RETURN_OO_05', quantiles=5, top_k=0, axvlines=('2020-01-01', '2024-01-01',))

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
    os.system(f'explorer.exe "{OUTPUT_PATH}"')
