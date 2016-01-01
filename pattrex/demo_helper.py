"""
@motjuste
"""
import numpy as np

# ## READING WHDATA.DAT #########################
WHDATA_PATH = "data/whData.dat"


def read_whdata():
    dt = np.dtype(
            [('w', np.float), ('h', np.float), ('g', 'S1')])  # g is byte-string

    data = np.loadtxt(WHDATA_PATH, dtype=dt, comments='#', delimiter=None)

    ws = np.array([d[0] for d in data])
    hs = np.array([d[1] for d in data])
    gs = np.array([d[2].decode('utf-8') for d in data])

    return ws, hs, gs
