'''
@motjuste
'''
import numpy as np


def all_positive_measurements(data, col_wise=True):
    # find indices for negative measurements
    neg_idx = np.where(data < 0)

    # if data is row/column wise, find unique rows/columns
    #   where any measurement is negative
    d = 1 if col_wise else 0
    neg_idx_unique = neg_idx[d]

    # delete row/column and return
    return np.delete(data, neg_idx_unique, d)
