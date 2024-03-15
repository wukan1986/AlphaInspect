"""
因子筛选
1. 先筛选出IC、IR满足要求的因子
2. 对多个因子的IC序列计算相关矩阵
    1. 多资产多因子的数值是三维的，无法计算相关矩阵。而它们的IC正好去了资产维度，降维成了二维、可以计算相关矩阵
    2. 单资产多因子的数值是二维的，可计算相关矩阵

References
----------
featuretools.selection.remove_highly_correlated_features
https://github.com/GauravSharma47/Drop-above-corr-threshold/blob/master/Drop_Above_correlation_Threshold.ipynb
"""
from typing import List, Tuple

import pandas as pd


def drop_above_corr_thresh(df: pd.DataFrame, thresh: float = 0.85) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    # This function returns list of columns to drop whose correlation is above specified threshold.
    corr = df.corr()
    cols = corr.columns.to_list()
    cols_to_drop = set()  # a set of columns which will be returned to the user to drop
    above_thresh_pairs = []  # pairs with correlation above specified threshold

    for i in range(len(cols)):
        for j in range(i, len(cols)):
            a, b = cols[i], cols[j]
            val = corr.iloc[i, j]
            if abs(val) > thresh and i != j:
                # if correlation is greater then threshold and columns are different
                above_thresh_pairs.append((a, b, val))
    # Now we'll compare the overall sum of absoulte correlation of each above threshold feature
    # with every other feature in the dataset
    # the feature with greater threshold will be added to cols_to_drop and ultimately dropped.
    for pair in above_thresh_pairs:
        a = abs(corr[pair[0]]).sum()
        b = abs(corr[pair[1]]).sum()
        if a > b:
            cols_to_drop.add(pair[0])
        else:
            cols_to_drop.add(pair[1])
    return list(cols_to_drop), above_thresh_pairs
