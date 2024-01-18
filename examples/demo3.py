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

ipynb_to_html('examples/template.ipynb',
              output='examples/demo3.html',
              no_input=True,
              no_prompt=False,
              open_browser=True,
              TEST1='123  456',
              TEST2='aaa  ccc')
