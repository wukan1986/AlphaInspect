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
                             factor=factor,
                             fwd_ret_1='RETURN_OO_1',
                             forward_return='RETURN_OO_5',
                             period=5)
    r"""
    https://github.com/python/cpython/issues/83595
    
    Exception ignored in: <function Pool.__del__ at 0x0000023C30726CA0>
    Traceback (most recent call last):
      File "D:\Users\Kan\miniconda3\envs\py311\Lib\multiprocessing\pool.py", line 271, in __del__
      File "D:\Users\Kan\miniconda3\envs\py311\Lib\multiprocessing\queues.py", line 371, in put
    AttributeError: 'NoneType' object has no attribute 'dumps'
    """
    return ret_code


if __name__ == '__main__':
    import multiprocessing

    # 没必要设置太大，因为部分计算使用的polars多线程，会将CPU跑满
    _map = multiprocessing.Pool(8).map

    factors = ['SMA_005', 'SMA_010', 'SMA_020']
    output = list(_map(func, factors))
    print(output)
