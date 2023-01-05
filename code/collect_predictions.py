
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
#opts = '_averaged-source_seitzman_nodes_average_runs_REST1_REST1_REST2_REST2-subs_651-params_FC_gm_FSL025_no_overlap_dt_flt01_001-beh_'
#opts = '_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695-beh_'
opts = '_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695_zscored-beh_'
#opts = '_averaged-source_Seitzman_nodes300_WM+CSF+GS_hcpaging_650_zscored-beh_'

# paths 
#in_path = Path('/data/project/impulsivity/Prediction_HCP/res/mean_accuracy')
#out_path = Path('/data/project/impulsivity/Prediction_HCP/res/collected')
in_path = Path('/data/project/impulsivity/prediction_simulations/res/mean_accuracy')
out_path = Path('/data/project/impulsivity/prediction_simulations/res/collected')

# which beh was simulated? excluding rel values.
# e.g.: 'interview_age_wnoise'
if beh == 'age_ridgeCV':
    beh_file = 'interview_age_wnoise'
    pipe = 'ridgeCV'
    # First load empirical results, then append all simulation res to it
    # ridgeCV_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695-beh_interview_age_interview_age-rseed_123456-cv_res.csv
    res = pd.read_csv(f'{in_path}/{pipe}{opts}interview_age_interview_age-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'age_ridgeCV_z':
    beh_file = 'interview_age_wnoise'
    pipe = 'ridgeCV_zscore'
    # First load empirical results, then append all simulation res to it
    # ridgeCV_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695-beh_interview_age_interview_age-rseed_123456-cv_res.csv
    res = pd.read_csv(f'{in_path}/ridgeCV{opts}interview_age_interview_age-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'age_ridgeCV_z_new':
    beh_file = 'interview_age_wnoise'
    pipe = 'ridgeCV_zscore'
    # First load empirical results, then append all simulation res to it
    # ridgeCV_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695-beh_interview_age_interview_age-rseed_123456-cv_res.csv
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}interview_age_interview_age-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'age_SVR_heuristic_z_new':
    beh_file = 'interview_age_wnoise'
    pipe = 'svr_heuristic_zscore'
    # First load empirical results, then append all simulation res to it
    # ridgeCV_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695-beh_interview_age_interview_age-rseed_123456-cv_res.csv
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}interview_age_interview_age-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'grip_ridge':
    beh_file = 'nih_tlbx_agecsc_dominant_wnoise'
    pipe = 'ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_motor_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]
elif beh == 'grip_ridge':
    beh_file = 'nih_tlbx_agecsc_dominant_wnoise'
    pipe = 'ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}nih_tlbx_agecsc_dominant_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]
elif beh == 'grip_ridgeCV_z':
    beh_file = 'HCP_A_motor_wnoise' #ridgeCV_zscore_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695-beh_HCP_A_motor_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv
    pipe = 'ridgeCV_zscore'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}HCP_A_motor_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'grip_ridgeCV_z_new':
    beh_file = 'HCP_A_motor_wnoise' #-beh_beh_HCP_A_cryst_wnoise_rel_095_80_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv
    pipe = 'ridgeCV_zscore'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}HCP_A_motor_nih_tlbx_agecsc_dominant-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'lswm_ridge':
    beh_file = 'age_corrected_standard_score_true_score_wnoise'
    pipe = 'ridge'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}beh_HCP_A_lswm_age_corrected_standard_score-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99]#[0.92,0.94,0.96]
elif beh == 'crycog_ridge':
    #ridge_averaged-source_seitzman_nodes_average_runs_REST1_REST1_REST2_REST2-subs_651-params_FC_gm_FSL025_no_overlap_dt_flt01_001-beh_nih_crycogcomp_ageadjusted_wnoise_rel_095_17_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv
    beh_file = 'nih_crycogcomp_ageadjusted_wnoise' # nih_crycogcomp_ageadjusted_wnoise_rel_085_95_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv
    pipe = 'ridgeCV'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}nih_crycogcomp_ageadjusted_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.85, 0.65]
elif beh == 'crycog_ridgeCV':
    beh_file = 'HCP_A_cryst_wnoise' #-beh_beh_HCP_A_cryst_wnoise_rel_095_80_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv
    pipe = 'ridgeCV'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}HCP_A_cryst_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'crycog_ridgeCV_z_new':
    beh_file = 'HCP_A_cryst_wnoise' #-beh_beh_HCP_A_cryst_wnoise_rel_095_80_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv
    pipe = 'ridgeCV_zscore'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}HCP_A_cryst_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'crycog_ridgeCV_z':
    beh_file = 'HCP_A_cryst_wnoise' #-beh_beh_HCP_A_cryst_wnoise_rel_095_80_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv
    pipe = 'ridgeCV_zscore'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}HCP_A_cryst_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'crycog_ridgeCV_exact_distribution':
    beh_file = 'HCP_A_cryst_wnoise'
    pipe = 'ridgeCV'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/{pipe}{opts}HCP_A_cryst_nih_crycogcomp_ageadjusted-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'total_ridgeCV_z_new':
    beh_file = 'HCP_A_total_wnoise'
    pipe = 'ridgeCV_zscore'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}HCP_A_total_nih_totalcogcomp_ageadjusted-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]

# which files
f_designator = pipe+opts+beh_file
print(f'Looking for: {f_designator}_rel_*')

for rel_i in reliabilities:
    rel_i_str = str(rel_i)
    print(rel_i_str)
    files = glob(f"{in_path}/pipe_{f_designator}_rel_{rel_i_str.replace('.','')}_*")

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
