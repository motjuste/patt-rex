'''
@motjuste
@kangcifong
'''
import numpy as np
from scipy.stats import norm


def fit_normal_distribution(data, x_pad=10):
    data_mean = np.mean(data)
    data_std = np.std(data)

    x = np.linspace(data.min()-x_pad, data.max()+x_pad, 100)
    y = norm.pdf(x, data_mean, data_std)
    return (data_mean, data_std, x, y)
