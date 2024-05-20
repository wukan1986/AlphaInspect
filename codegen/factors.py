import sys

from expr_codegen.tool import codegen_exec
from polars_ta.prefix.wq import ts_max, ts_min, ts_mean, ts_std_dev, ts_returns


def _code_block_1():
    # 远期收益率,由于平移过,含未来数据，只能用于打标签，不能用于训练
    HHV_005 = ts_max(HIGH, 5) / CLOSE
    HHV_010 = ts_max(HIGH, 10) / CLOSE
    HHV_020 = ts_max(HIGH, 20) / CLOSE
    HHV_060 = ts_max(HIGH, 60) / CLOSE
    HHV_120 = ts_max(HIGH, 120) / CLOSE

    LLV_005 = ts_min(LOW, 5) / CLOSE
    LLV_010 = ts_min(LOW, 10) / CLOSE
    LLV_020 = ts_min(LOW, 20) / CLOSE
    LLV_060 = ts_min(LOW, 60) / CLOSE
    LLV_120 = ts_min(LOW, 120) / CLOSE

    SMA_005 = ts_mean(CLOSE, 5) / CLOSE
    SMA_010 = ts_mean(CLOSE, 10) / CLOSE
    SMA_020 = ts_mean(CLOSE, 20) / CLOSE
    SMA_060 = ts_mean(CLOSE, 60) / CLOSE
    SMA_120 = ts_mean(CLOSE, 120) / CLOSE

    STD_005 = ts_std_dev(CLOSE, 5) / CLOSE
    STD_010 = ts_std_dev(CLOSE, 10) / CLOSE
    STD_020 = ts_std_dev(CLOSE, 20) / CLOSE
    STD_060 = ts_std_dev(CLOSE, 60) / CLOSE
    STD_120 = ts_std_dev(CLOSE, 120) / CLOSE

    ROCP_001 = ts_returns(CLOSE, 1)
    ROCP_003 = ts_returns(CLOSE, 3)
    ROCP_005 = ts_returns(CLOSE, 5)
    ROCP_010 = ts_returns(CLOSE, 10)
    ROCP_020 = ts_returns(CLOSE, 20)
    ROCP_060 = ts_returns(CLOSE, 60)
    ROCP_120 = ts_returns(CLOSE, 120)


df = None
df = codegen_exec(df, _code_block_1, output_file=sys.stdout)
