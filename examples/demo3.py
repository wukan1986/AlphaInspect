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
                             forward_return='RETURN_OO_05')

    return ret_code


if __name__ == '__main__':
    import multiprocessing

    factors = ['SMA_005', 'SMA_010', 'SMA_020']

    # 没必要设置太大，因为部分计算使用的polars多线程，会将CPU跑满
    with multiprocessing.Pool(8) as pool:
        output = list(pool.map(func, factors))
        print(output)
