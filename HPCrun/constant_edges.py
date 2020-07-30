import numpy as np
import minkowski_tools as mt


for edges in [2, 4, 8, 16]:

    savename =f'constant_edges{edges}.txt'

    for _ in range(3):
        
        # ns = np.arange(200, 10200, 200)
        # ps = [p]*len(ns)
        # rs = [0.2]*len(ps)
        
        ps = np.arange(0.2, 3, 0.05)
        n = 20000
        ns = [n]*len(ps)
        rs = [mt.norm_kernel_2D(p, edges/n) for n, p in zip(ns, ps)]
        
        
        mt.separate_simulations(ns, ps, rs, savename)
