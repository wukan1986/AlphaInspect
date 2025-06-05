"""
问题：明日涨跌停是否要过滤?

1. 回测分层累计收益时应当过滤掉涨跌停
    因为开盘前预测出股票后，开盘后才发现涨跌停无法交易影响曲线
2. IC计算，是否要过滤掉涨跌停呢？个人认为不能过滤掉
    因为IC用于评价预测能力，能不能交易不是它能实现的
3. 机器学习。应当保留涨跌停。原因与IC一样
    涨跌停的记录全删了后，失去学习涨跌停相关信息的机会。

有不同观点的朋友可以提issue

"""
import base64
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple, Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger  # noqa
from matplotlib import pyplot as plt

from alphainspect import _QUANTILE_, _DATE_
from alphainspect.ic import calc_ic
from alphainspect.plotting import plot_hist, plot_heatmap_monthly_mean, plot_ts
from alphainspect.portfolio import calc_cum_return_by_quantile, plot_quantile_portfolio
from alphainspect.turnover import calc_auto_correlation, calc_quantile_turnover, plot_factor_auto_correlation, \
    plot_turnover_quantile
from alphainspect.utils import with_factor_quantile, with_factor_top_k

html_template = """
<html>
<head>
<style>
table { border-collapse: collapse;}
img {border: 1px solid;}
</style>
</head>
<body>
{{body}}
</body>
</html>
"""


def fig_to_img(fig, format: str = "png") -> str:
    """图片转HTML字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format=format)
    return '<img src="data:image/{};base64,{}" />'.format(format,
                                                          base64.b64encode(buf.getvalue()).decode())


def ipynb_to_html(template: str, output: str = None,
                  no_input: bool = False, no_prompt: bool = False, execute: bool = True,
                  timeout: int = 120,
                  open_browser: bool = True,
                  **kwargs) -> int:
    """将`ipynb`导出成`HTML`格式。生成有点慢，也许自己生成html更好

    Parameters
    ----------
    template: str
        模板，ipynb格式
    output:str`
        输出html
    no_input: bool
        无输入
    no_prompt: bool
        无提示
    execute: bool
        是否执行
    timeout: int
        执行超时
    open_browser: bool
        是否打开浏览器
    kwargs: dict
        环境变量。最终转成大写，所以与前面的参数不会冲突

    """
    template = Path(template).absolute()
    template = str(template)
    if not template.endswith('.ipynb'):
        raise ValueError('template must be a ipynb file')

    if output is None:
        output = template.replace('.ipynb', '.html')
    output = Path(output).absolute()

    no_input = '--no-input' if no_input else ''
    no_prompt = '--no-prompt' if no_prompt else ''
    execute = '--execute' if execute else ''
    command = f'jupyter nbconvert "{template}" --to=html --output="{output}" {no_input} {no_prompt} {execute} --allow-errors --ExecutePreprocessor.timeout={timeout}'

    # 环境变量名必须大写，值只能是字符串
    kwargs = {k.upper(): str(v) for k, v in kwargs.items()}
    # 担心环境变量副作用，同时跑多个影响其它进程，所以不用 os.environ
    # os.environ.update(kwargs)

    if os.name == 'nt':
        cmds = [f'set {k}={v}' for k, v in kwargs.items()] + [command]
        # commands = ' & '.join(cmds)
    else:
        cmds = [f'export {k}={v}' for k, v in kwargs.items()] + [command]
        # commands = ' ; '.join(cmds)

    commands = '&&'.join(cmds)

    # print('environ:', kwargs)
    # print('command:', command)
    print('system:', commands)

    ret = os.system(commands)
    if ret == 0 and open_browser:
        # print(f'open {output}')
        os.system(f'"{output}"')
    return ret


def create_2x2_sheet(df: pl.DataFrame,
                     factor: str,
                     fwd_ret_1: str,
                     *,
                     factor_quantile: str = _QUANTILE_,
                     figsize=(12, 9),
                     axvlines: Sequence[str] = ()) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """画2*2的图表。含IC时序、IC直方图、IC热力图、累积收益图

    Parameters
    ----------
    df
    factor
    fwd_ret_1:str
        用于记算累计收益的1期远期收益率
    factor_quantile:str
    figsize

    axvlines

    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # 画IC信息
    # logger.info('计算IC')
    df_ic = calc_ic(df, [factor], [fwd_ret_1])
    col = df_ic.columns[1]
    ic_dict = plot_ts(df_ic, col, axvlines=axvlines, ax=axes[0])
    plot_heatmap_monthly_mean(df_ic, col, ax=axes[1])

    # 画因子直方图
    hist_dict = plot_hist(df, factor, ax=axes[2])

    # 画累计收益
    ret, cum, avg, std = calc_cum_return_by_quantile(df, fwd_ret_1, factor_quantile)
    plot_quantile_portfolio(cum, fwd_ret_1, axvlines=axvlines, ax=axes[3])

    fig.tight_layout()

    return fig, ic_dict, hist_dict, cum, avg, std


def create_1x3_sheet(df: pl.DataFrame,
                     factor: str,
                     fwd_ret_1: str,
                     *,
                     factor_quantile: str = _QUANTILE_,
                     figsize=(12, 4),
                     axvlines: Sequence[str] = ()) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """画2*2的图表。含IC时序、IC直方图、IC热力图、累积收益图

    Parameters
    ----------
    df
    factor
    fwd_ret_1:str
        用于记算累计收益的1期远期收益率
    factor_quantile:str
    figsize
    axvlines

    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # 画IC信息
    # logger.info('计算IC')
    df_ic = calc_ic(df, [factor], [fwd_ret_1])
    col = df_ic.columns[1]
    ic_dict = plot_ts(df_ic, col, axvlines=axvlines, ax=axes[0])

    # 画累计收益
    ret, cum, avg, std = calc_cum_return_by_quantile(df, fwd_ret_1, factor_quantile)
    plot_quantile_portfolio(cum, fwd_ret_1, axvlines=axvlines, ax=axes[1])

    # 画因子直方图
    hist_dict = plot_hist(df, factor, ax=axes[2])

    fig.tight_layout()

    return fig, ic_dict, hist_dict, cum, avg, std


def create_3x2_sheet(df: pl.DataFrame,
                     factor: str,
                     fwd_ret_1: str,
                     *,
                     factor_quantile: str = _QUANTILE_,
                     periods: Sequence[int] = (1, 5, 10, 20),
                     figsize=(12, 14),
                     axvlines: Sequence[str] = ()) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """画2*3图

    Parameters
    ----------
    df
    factor
    fwd_ret_1:str
        用于记算累计收益的1期远期收益率
    periods:
        换手率，多期比较
    factor_quantile:str
    axvlines

    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # 画IC信息
    # logger.info('计算IC')
    df_ic = calc_ic(df, [factor], [fwd_ret_1])
    col = df_ic.columns[1]
    ic_dict = plot_ts(df_ic, col, axvlines=axvlines, ax=axes[0, 0])
    hist_dict = plot_hist(df_ic, col, ax=axes[0, 1])
    plot_heatmap_monthly_mean(df_ic, col, ax=axes[1, 0])

    # 画净值曲线
    # logger.info('计算累计收益')
    ret, cum, avg, std = calc_cum_return_by_quantile(df, fwd_ret_1, factor_quantile)
    plot_quantile_portfolio(cum, fwd_ret_1, axvlines=axvlines, ax=axes[1, 1])

    # 画换手率
    # logger.info('计算换手率')
    df_auto_corr = calc_auto_correlation(df, factor, periods=periods)
    df_turnover = calc_quantile_turnover(df, periods=periods, factor_quantile=factor_quantile)
    plot_factor_auto_correlation(df_auto_corr, axvlines=axvlines, ax=axes[2, 0])

    q_min, q_max = df_turnover[factor_quantile].min(), df_turnover[factor_quantile].max()
    plot_turnover_quantile(df_turnover, quantile=q_max, factor_quantile=factor_quantile, periods=periods,
                           axvlines=axvlines, ax=axes[2, 1])

    fig.tight_layout()

    return fig, ic_dict, hist_dict, cum, avg, std


def report_html(name: str, factors, df, output: str,
                *,
                fwd_ret_1: str = 'RETURN_OO_05', quantiles: int = None, top_k: int = None,
                axvlines: tuple[str, str] = ('2020-01-01', '2024-01-01',)):
    tbl = {}
    df_mean = {}
    df_std = {}
    df_last = {}
    imgs = []

    for factor in factors:
        if quantiles and quantiles > 0:
            df = with_factor_quantile(df, factor, quantiles=quantiles, by=[_DATE_], factor_quantile=f'_fq_{factor}')
            continue
        if top_k and top_k > 0:
            df = with_factor_top_k(df, factor, top_k=top_k, by=[_DATE_], factor_quantile=f'_fq_{factor}')
            continue

    for factor in factors:
        fig, ic_dict, hist_dict, cum, avg, std = create_1x3_sheet(df, factor, fwd_ret_1,
                                                                  factor_quantile=f'_fq_{factor}', axvlines=axvlines)

        cum = cum.to_pandas().set_index('date').iloc[-1]
        avg = avg.to_pandas().set_index('date').iloc[-1]
        std = std.to_pandas().set_index('date').iloc[-1]

        df_last[factor] = cum
        df_mean[factor] = avg
        df_std[factor] = std

        s2 = {'monotonic': np.sign(cum.diff()).sum()}
        s3 = pd.Series(s2 | ic_dict | hist_dict)
        tbl[factor] = pd.concat([cum, s3])
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
    tpl = html_template.replace('{{body}}', f'{datetime.now()}\n{txt1}\n{txt2}')

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    with open(output / f'{name}.html', "w", encoding="utf-8") as f:
        f.write(tpl)
