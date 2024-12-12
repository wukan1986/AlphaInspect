"""
Notebook报告

会生成多个html文件。效果好
1. 多因子比较时要翻看多个文件比较辛苦
2. 每个进程要重新加载数据，比较占用IO资源

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
from alphainspect.reports import ipynb_to_html


def func(factor):
    ret_code = ipynb_to_html('examples/template.ipynb',
                             output=f'examples/{factor}.html',
                             no_input=True,
                             no_prompt=False,
                             open_browser=False,
                             # 以下参数转成环境变量自动变成大写
                             factor=factor,
                             fwd_ret_1='RETURN_OO_05',
                             )

    return ret_code


if __name__ == '__main__':
    import multiprocessing

    factors = ['SMA_005', 'SMA_010', 'SMA_020']

    # 没必要设置太大，因为部分计算使用的polars多线程，会将CPU跑满
    with multiprocessing.Pool(8) as pool:
        output = list(pool.map(func, factors))
        print(output)
