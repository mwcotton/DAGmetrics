import numpy as np
import minkowskitools as mt
import os

home = os.environ["HOME"]

repeats = 400

for n in [5000, 10000, 20000]:

    ps = np.arange(2, 3.1, 0.2)
    ns = [n]*len(ps)

    mt.separate_perc_r(ns, ps, f'{home}/perc_r/perc_r{n}another.txt', repeats)
