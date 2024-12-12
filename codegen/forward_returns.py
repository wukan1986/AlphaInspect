import sys

from expr_codegen import codegen_exec


def _code_block_1():
    # 远期收益率,由于平移过,含未来数据，只能用于打标签
    _OC_01 = CLOSE[-1] / OPEN[-1]
    _CC_01 = CLOSE[-1] / CLOSE
    _CO_01 = OPEN[-1] / CLOSE
    _OO_01 = OPEN[-2] / OPEN[-1]

    _OO_02 = OPEN[-3] / OPEN[-1]
    _OO_05 = OPEN[-6] / OPEN[-1]
    _OO_10 = OPEN[-11] / OPEN[-1]

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
