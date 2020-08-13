import minkowskitools as mt
import numpy as np
import os

home = os.environ["HOME"]

fileName = f'{home}/perc_n.txt'

# fileName = '/rds/general/user/mwc116/home/perc_n/output.txt'

ps = []
rs = []

for p in np.arange(0.4, 3.1, 0.2):
    for r in [0.05, 0.06, 0.07, 0.08, 0.1, .12, .15, .2]:
        ps.append(p)
        rs.append(r)

mt.ensemble_perc_n(fileName, ps, rs, repeats=400)