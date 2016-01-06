"""
@motjuste
"""
import numpy as np


def only_all_positive(data, colwise_data, return_neg_idx=False):
    # find indices for negative measurements
    neg_idx = np.where(data < 0)

    # if data is row/column wise, find unique rows/columns
    #   where any measurement is negative
    d = 1 if colwise_data else 0
    neg_idx_unique = neg_idx[d]

    # delete row/column and return
    new_data = np.delete(data, neg_idx_unique, d)

    if return_neg_idx:
        return new_data, neg_idx_unique
    else:
        return new_data


def split_data(data, colwise_data, field_idx, field_values):
    splits = []
    if colwise_data:
        for value in field_values:
            data_idx = np.where(data[field_idx, :] == value)[0]
            splits.append(data[:, data_idx])

    else:
        for value in field_values:
            data_idx = np.where(data[:, field_idx] == value)[0]
            splits.append(data[data_idx, :])

    return splits
