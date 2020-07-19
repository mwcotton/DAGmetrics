import numpy as np
import minkowski_tools as mt

savename ='outputs/4000_40bigrange.txt'

n = 4000
ps = np.arange(.2, 3.01, 0.01)
ns = [n]*len(ps)
rs = mt.norm_kernel_2D(ps, 40/n)
mt.separate_simulations(ns, ps, rs, savename)
