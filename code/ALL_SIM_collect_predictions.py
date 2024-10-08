
'/data/project/impulsivity/prediction_simulations/res/mean_accuracy/pipe_ridgeCV_averaged-source_all_simulated_vars_X-beh_simulated_wnoise_rel_099498743710662_95_simulated-rseed_123456-cv_res.csv'
#ridgeCV_averaged-source_all_simulated_vars_X-beh_simulated_wnoise_rel
#ridgeCV_averaged-source_all_simulated_vars_X-beh_simulated_wnoise_rel_*

# Collect predictions
from glob import glob
from pathlib import Path
import sys
import pandas as pd
import numpy as np


# sample used
#beh = 'crycog_ridgeCV'
beh = sys.argv[1]
n = sys.argv[2]

opts = '_averaged-source_all_simulated_vars_X-beh_'

# paths 
#in_path = Path('/data/project/impulsivity/Prediction_HCP/res/mean_accuracy')
#out_path = Path('/data/project/impulsivity/Prediction_HCP/res/collected')
in_path = Path('/data/project/impulsivity/prediction_simulations/res/mean_accuracy')
out_path = Path('/data/project/impulsivity/prediction_simulations/res/collected')

# which beh was simulated? excluding rel values.
# e.g.: 'interview_age_wnoise'
if beh == 'ridgeCV':
    beh_file = 'simulated_wnoise' #-beh_beh_HCP_A_cryst_wnoise_rel_095_80_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv
    pipe = 'ridgeCV'
    # First load empirical results, then append all simulation res to it
    # pipe_ridgeCV_averaged-source_all_simulated_vars_X-beh_all_simulated_vars_Y_simulated-rseed_123456-cv_res.csv
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}all_simulated_vars_Y_simulated-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]


# which files
f_designator = pipe+opts+beh_file
print(f'Looking for: {f_designator}_rel_*')

for rel_i in reliabilities:
    rel_i_str = np.sqrt(rel_i)
    rel_i_str = np.round(rel_i_str,decimals=15)
    rel_i_str = str(rel_i_str)
    print(f'{rel_i} == {rel_i_str}')
    files = glob(f"{in_path}/pipe_{f_designator}_rel_{rel_i_str.replace('.','')}*")
    print(len(files))

    for f_i in files:
        f = pd.read_csv(f_i)
        f['reliability'] = rel_i
        res = res.append(f, ignore_index=True)

# Save
out_path.mkdir(parents=True, exist_ok=True)
out_file = out_path / f'{pipe}{opts}{beh_file}_all.csv'
res.to_csv(out_file, index=False)


# for rel_i in range(49):
#     rel_i_str = str(rel_i)
#     f_i = f"{in_path}/{f_designator}_{rel_i+2}_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv"
#     f = pd.read_csv(f_i)
#     res = res.append(f, ignore_index=True)
