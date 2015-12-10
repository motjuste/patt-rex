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


def aitchison_norm(xyz):
    lxyz = np.log(xyz)
    lgxyz = np.mean(lxyz, axis=0)

    # test show tiling not required
#     if len(lxyz.shape) > 1:
#         lgxyz = np.tile(lgxyz, (lxyz.shape[0], 1, 1, 1))

    diff = lxyz - lgxyz
    return np.sqrt(np.sum(diff * diff, axis=0))
