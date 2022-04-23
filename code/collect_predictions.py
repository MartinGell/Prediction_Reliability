
# Collect predictions
from glob import glob
from pathlib import Path
import sys
import pandas as pd
import numpy as np


# sample used
#beh = 'grip'
beh = sys.argv[1]
opts = '_averaged-source_seitzman_nodes_average_runs_REST1_REST1_REST2_REST2-subs_651-params_FC_gm_FSL025_no_overlap_dt_flt01_001-beh_'

# paths 
in_path = Path('/data/project/impulsivity/Prediction_HCP/res/mean_accuracy')
#in_path = Path('/home/mgell/Work/reliability/input/mean_accuracy/mean_accuracy')
out_path = Path('/data/project/impulsivity/Prediction_HCP/res/collected')
#out_path = Path('/home/mgell/Work/reliability/input')

# which beh was simulated? excluding rel values.
# e.g.: 'interview_age_wnoise'
if beh == 'age':
    beh_file = 'interview_age_wnoise'
    pipe = 'ridge_simple'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_age_interview_age-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'age_true':
    #ridge_simple_averaged-source_seitzman_nodes_average_runs_REST1_REST1_REST2_REST2-subs_651-params_FC_gm_FSL025_no_overlap_dt_flt01_001-beh_interview_age_wnoise_rel_099_11_interview_age-rseed_123456-cv_res.csv
    beh_file = 'interview_age_wnoise'
    pipe = 'ridge_simple'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_age_interview_age-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99]
elif beh == 'grip_ridge':
    beh_file = 'nih_tlbx_agecsc_dominant_wnoise'
    pipe = 'ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_motor_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]
elif beh == 'grip_ridge':
    beh_file = 'nih_tlbx_agecsc_dominant_450_1_wnoise'
    pipe = 'ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_motor_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]
elif beh == 'grip_true_ridge':
    beh_file = 'nih_tlbx_agecsc_dominant_true_score_wnoise'
    pipe = 'ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_motor_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99]#[0.92,0.94,0.96]
elif beh == 'lswm_ridge':
    beh_file = 'age_corrected_standard_score_true_score_wnoise'
    pipe = 'ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_lswm_age_corrected_standard_score-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99]#[0.92,0.94,0.96]
elif beh == 'crycog_ridge':
    #ridge_averaged-source_seitzman_nodes_average_runs_REST1_REST1_REST2_REST2-subs_651-params_FC_gm_FSL025_no_overlap_dt_flt01_001-beh_nih_crycogcomp_ageadjusted_wnoise_rel_095_17_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv
    beh_file = 'nih_crycogcomp_ageadjusted_wnoise'
    pipe = 'ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_cryst_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]
elif beh == 'crycog_true_ridge':
    #ridge_averaged-source_seitzman_nodes_average_runs_REST1_REST1_REST2_REST2-subs_651-params_FC_gm_FSL025_no_overlap_dt_flt01_001-beh_nih_crycogcomp_ageadjusted_true_score_wnoise_rel_095_17_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv
    beh_file = 'nih_crycogcomp_ageadjusted_true_score_wnoise'
    pipe = 'ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_cryst_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99]

# which files
f_designator = pipe+opts+beh_file

for rel_i in reliabilities:
    rel_i_str = str(rel_i)
    files = glob(f"{in_path}/{f_designator}_rel_{rel_i_str.replace('.','')}_*")

    for f_i in files:
        f = pd.read_csv(f_i)
        f['reliability'] = rel_i
        res = res.append(f, ignore_index=True)

# Save
out_file = out_path / f'{pipe}{opts}{beh_file}_all.csv'
res.to_csv(out_file, index=False)
