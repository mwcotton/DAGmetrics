import numpy as np
import minkowski_tools as mt


# for edges in [5, 10, 20, 40]:

savename =f'outputs/dimensionrun.txt'

for p in [0.8, 1.2]:

    ns = np.arange(200, 10200, 200)
    # ps = np.arange(0.4, 3, 0.05)
    ps = [p]*len(ns)
    # n = 4000
    # ns = [n]*len(ps)
    rs = [mt.norm_kernel_2D(p, 5/n) for n, p in zip(ns, ps)]
    # rs = [0.2]*len(ps)
    mt.separate_simulations(ns, ps, rs, savename)
