import sys

from expr_codegen.tool import codegen_exec
from polars_ta.prefix.wq import ts_delay


def _code_block_1():
    # 远期收益率,由于平移过,含未来数据，只能用于打标签
    _OC_01 = ts_delay(CLOSE, -1) / ts_delay(OPEN, -1)
    _CC_01 = ts_delay(CLOSE, -1) / CLOSE
    _CO_01 = ts_delay(OPEN, -1) / CLOSE
    _OO_01 = ts_delay(OPEN, -2) / ts_delay(OPEN, -1)

    _OO_02 = ts_delay(OPEN, -3) / ts_delay(OPEN, -1)
    _OO_05 = ts_delay(OPEN, -6) / ts_delay(OPEN, -1)
    _OO_10 = ts_delay(OPEN, -11) / ts_delay(OPEN, -1)

    # 一期收益率
    RETURN_OC_01 = _OC_01 - 1
    RETURN_CC_01 = _CC_01 - 1
    RETURN_CO_01 = _CO_01 - 1
    RETURN_OO_01 = _OO_01 - 1

    # 算术平均
    RETURN_OO_02 = (_OO_02 - 1) / 2
    RETURN_OO_05 = (_OO_05 - 1) / 5
    RETURN_OO_10 = (_OO_10 - 1) / 10

    # 几何平均
    RETURN_OO_02 = _OO_02 ** (1 / 2) - 1
    RETURN_OO_05 = _OO_05 ** (1 / 5) - 1
    RETURN_OO_10 = _OO_10 ** (1 / 10) - 1


df = None
df = codegen_exec(df, _code_block_1, output_file=sys.stdout)
