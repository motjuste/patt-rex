'''
@motjuste : 22-Nov-2015

- Code relevant to Task 1.4
'''
import numpy as np
import matplotlib.pyplot as plt


def lp_unit_circle(p=2):
    x = np.linspace(0, 1, 100)
    y = np.power((1 - np.power(x, p)), 1/p)

    return (x, y)
