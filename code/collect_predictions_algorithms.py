# Collect predictions
from glob import glob
from pathlib import Path
import sys
import pandas as pd
import numpy as np


# Paths
in_path = Path('/data/project/impulsivity/prediction_simulations/res/mean_accuracy')
out_path = Path('/data/project/impulsivity/prediction_simulations/res/collected')

all_res = []

# All algs to collect
# svr_L2 time = 9,5,5
# svr time = 23.1,17.5,17.5
# kridge time = 91.2,66,66
# ridge = 4,3.3,3.3

patterns = ['svr_L2', 'svr_heuristic_zscore', 'kridge_averaged']#, 'ridge...']
timing   = [[9,5,5],[23.1,17.5,17.5],[91.2,66,66]]#,[4,3.3,3.3]]
i = 0

for pattern in patterns:

    files = glob(f"{in_path}/pipe_{pattern}*")
    alg_res = []

    for file in files:
        res = pd.read_csv(file)
        name = file.split('/')
        res['alg'] = pattern
        res['beh'] = name[-1].split('-')[2]
        alg_res.append(res)

    alg_res = pd.concat(alg_res, axis=0)
    alg_res = pd.concat([alg_res.reset_index(),pd.DataFrame(timing[i], columns=['compute_time'])], axis=1)
    all_res.append(alg_res)
    i=i+1

all_res = pd.concat(all_res, axis=0)

all_res.to_csv(f"{out_path}/empirical_behs_all_algs.csv", index=False)