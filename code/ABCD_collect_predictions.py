
# Collect predictions
from glob import glob
from pathlib import Path
import sys
from unittest import skip
import pandas as pd
import numpy as np


# sample used
pipe = sys.argv[1]

#opts = '_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_'
#opts = '_averaged-source_Schaefer400x17_WM+CSF+GS_hcpya_771_zscored-beh_'
opts = '_averaged-source_HCP2016FreeSurferSubcortical_abcd_baselineYear1Arm1_rest_3517_zscored-beh_'

# paths 
in_path = Path('/data/project/impulsivity/prediction_simulations/res/mean_accuracy')
out_path = Path('/data/project/impulsivity/prediction_simulations/res/collected')

# which beh was simulated? excluding rel values.
# e.g.: 'interview_age_wnoise'
if pipe == 'ridgeCV_z':
    beh_file = 'ABCD_beh_all'
    pipe = 'ridgeCV_zscore'
    first = 'lmt_scr_avg_rt'
    # First load empirical results, then append all simulation res to it
    # ABCD_beh_all_tfmri_nb_all_beh_ctotal_mrt-rseed_123456-cv_res.csv
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}{beh_file}_lmt_scr_avg_rt-rseed_123456-cv_res.csv')
    res['beh'] = first
    behs = pd.read_table('/data/project/impulsivity/prediction_simulations/code/opts/ABCD_behs2predict.txt', header=None)
elif pipe == 'ridgeCV_z_conf_removed':
    beh_file = 'ABCD_beh_all'
    pipe = 'ridgeCV_zscore_confound_removal_wcategorical'
    first = 'lmt_scr_avg_rt'
    # First load empirical results, then append all simulation res to it
    #pipe_ridgeCV_zscore_confound_removal_wcategorical_averaged-source_HCP2016FreeSurferSubcortical_abcd_baselineYear1Arm1_rest_3517_zscored-beh_ABCD_beh_all_tfmri_rec_all_beh_neutf_dp-rseed_123456-cv_res.csv
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}{beh_file}_lmt_scr_avg_rt-rseed_123456-cv_res.csv')
    res['beh'] = first
    behs = pd.read_table('/data/project/impulsivity/prediction_simulations/code/opts/ABCD_behs2predict.txt', header=None)


# which files
f_designator = pipe+opts+beh_file
print(f'Looking for: {f_designator}*')

for beh_i in behs[0]:

    beh_i_str = str(beh_i)
    print(beh_i_str)
    files = glob(f"{in_path}/pipe_{f_designator}_{beh_i_str}-*")

    for f_i in files:
        f = pd.read_csv(f_i)
        f['beh'] = beh_i_str
        res = res.append(f, ignore_index=True)

# Save
out_path.mkdir(parents=True, exist_ok=True)
out_file = out_path / f'{pipe}{opts}{beh_file}_all_behs.csv'
res.to_csv(out_file, index=False)


# for rel_i in range(49):
#     rel_i_str = str(rel_i)
#     f_i = f"{in_path}/{f_designator}_{rel_i+2}_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv"
#     f = pd.read_csv(f_i)
#     res = res.append(f, ignore_index=True)
