import minkowskitools as mt
import numpy as np
import os

home = os.environ["HOME"]

fileName = f'{home}/lim_scale/lim_scale.txt'

# fileName = '/rds/general/user/mwc116/home/perc_n/output.txt'

ps = []
ns = []
rs = []
r = 0.05
for n in np.arange(1000, 21000, 1000):    
    for p in [0.8, 1.2]:
        ps.append(p)
        ns.append(n)
        rs.append(r)

mt.all_paths_bulk(fileName, ps, ns, rs, repeats=10)