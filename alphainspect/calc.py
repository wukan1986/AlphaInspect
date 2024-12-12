import pandas as pd
import polars as pl

from alphainspect import _DATE_


def calc_mean(df: pl.DataFrame) -> pl.DataFrame:
    """计算IC的均值"""
    return df.select(pl.exclude(_DATE_).mean())


def calc_ir(df: pl.DataFrame) -> pl.DataFrame:
    """计算ir,需保证没有nan，只有null"""
    return df.select(pl.exclude(_DATE_).mean() / pl.exclude(_DATE_).std(ddof=0))


def calc_corr(df: pl.DataFrame) -> pd.DataFrame:
    """由于numpy版不能很好的处理空值，所以用pandas版"""
    return df.to_pandas().corr(method="pearson")
