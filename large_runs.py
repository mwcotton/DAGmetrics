import numpy as np
import minkowski_tools as mt

savename ='outputs/quick_run.txt'

for _ in range(10):
    # ns = np.arange(200, 10200, 200)
    ps = np.arange(0.4, 3, 0.05)
    n = 1000
    ns = [n]*len(ps)
    # rs = [mt.norm_kernel_2D(p, 5/n) for n, p in zip(ns, ps)]
    rs = [0.2]*len(ps)
    mt.separate_simulations(ns, ps, rs, savename)
