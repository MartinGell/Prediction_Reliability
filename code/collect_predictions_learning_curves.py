
# Collect predictions
from glob import glob
from pathlib import Path
import sys
import pandas as pd
import numpy as np


# sample used
#beh = 'grip'
beh = sys.argv[1]
opts = '_learning_curve-source_seitzman_nodes_average_runs_REST1_REST1_REST2_REST2-subs_651-params_FC_gm_FSL025_no_overlap_dt_flt01_001-beh_'

# paths 
in_path = Path('/data/project/impulsivity/Prediction_HCP/res/learning_curve')
#in_path = Path('/home/mgell/Work/reliability/input/mean_accuracy/mean_accuracy')
out_path = Path('/data/project/impulsivity/Prediction_HCP/res/learning_curve_collected')
#out_path = Path('/home/mgell/Work/reliability/input')

# which beh was simulated? excluding rel values.
# e.g.: 'interview_age_wnoise'
if beh == 'grip_ridge_test':
    #test_ridge_learning_curve-source_seitzman_nodes_average_runs_REST1_REST1_REST2_REST2-subs_651-params_FC_gm_FSL025_no_overlap_dt_flt01_001-beh_beh_HCP_A_motor_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv
    beh_file = 'nih_tlbx_agecsc_dominant_wnoise'
    pipe = 'test_ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_motor_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    res_mean = pd.DataFrame(np.mean(res,0)).transpose()
    res_sd = pd.DataFrame(np.std(res,0)).transpose()
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]
elif beh == 'grip_true_ridge':
    beh_file = 'nih_tlbx_agecsc_dominant_true_score_wnoise'
    pipe = 'test_ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_motor_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    res_mean = pd.DataFrame(np.mean(res,0)).transpose()
    res_sd = pd.DataFrame(np.std(res,0)).transpose()
    reliabilities = [0.99]#[0.92,0.94,0.96]
elif beh == 'grip_ridge_sample':
    beh_file = 'nih_tlbx_agecsc_dominant_450_1_wnoise'
    pipe = 'ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_motor_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]


# which files
f_designator = pipe+opts+beh_file

for rel_i in reliabilities:
    rel_i_str = str(rel_i)
    files = glob(f"{in_path}/{f_designator}_rel_{rel_i_str.replace('.','')}_*")

    for f_i in files:
        f = pd.read_csv(f_i)
        f['reliability'] = rel_i
        res_mean = res_mean.append(pd.DataFrame(np.mean(f,0)).transpose(), ignore_index=True)
        res_sd = res_sd.append(pd.DataFrame(np.std(f,0)).transpose(), ignore_index=True)

# Save
out_path.mkdir(parents=True, exist_ok=True)
out_file = out_path / f'{pipe}{opts}{beh_file}_mean_all.csv'
res_mean.to_csv(out_file, index=False)

out_file = out_path / f'{pipe}{opts}{beh_file}_sd_all.csv'
res_sd.to_csv(out_file, index=False)