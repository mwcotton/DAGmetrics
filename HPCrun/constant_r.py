import numpy as np
import minkowski_tools as mt


# for edges in [5, 10, 20, 40]:

savename ='constant_r.txt'

for _ in range(5):
    
    # ns = np.arange(200, 10200, 200)
    # ps = [p]*len(ns)
    # rs = [mt.norm_kernel_2D(p, 5/n) for n, p in zip(ns, ps)]


    ps = np.arange(0.2, 3, 0.05)
    n = 10000
    ns = [n]*len(ps)
    rs = [0.1]*len(ps)
    
    mt.separate_simulations(ns, ps, rs, savename)
