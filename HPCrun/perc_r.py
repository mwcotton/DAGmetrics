import numpy as np
import minkowski_tools as mt

savename = 'perc_r.txt'

repeats = 10000

for n in [5000, 10000, 20000]:

    ps = np.arange(0.4, 3.1, 0.2)
    ns = [n]*len(ps)

    mt.separate_perc_r(ns, ps, f'perc_r{n}.txt', repeats)
