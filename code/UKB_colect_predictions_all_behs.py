
# Collect predictions
from glob import glob
from pathlib import Path
import sys
from unittest import skip
import pandas as pd
import numpy as np


# sample used
pipe = sys.argv[1]

#opts = '_averaged-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_'
opts = '_averaged-source_Schaefer400x17_nodenoise_UKB_3353_z-beh_'


# paths 
in_path = Path('/data/project/ukb_reliability_in_prediction/prediction_UKB/res/mean_accuracy')
out_path = Path('/data/project/ukb_reliability_in_prediction/prediction_UKB/res/collected')

# which beh was simulated? excluding rel values.
# e.g.: 'interview_age_wnoise'
if pipe == 'ridgeCV_z':
    beh_file = 'UKB_1919_subs_FC_all_cogs'
    pipe = 'ridgeCV_zscore'
    first = 'MT_B_duration_to_complete-3.0'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}{beh_file}_TMT_B_duration_to_complete-3.0-rseed_123456-cv_res.csv')
    res['beh'] = first
    behs = pd.read_table('/data/project/ukb_reliability_in_prediction/prediction_UKB/code/opts/UKB_behs_all2.txt', header=None)
elif pipe == 'ridgeCV_zscore_confound_removal_wcategorical':
    #pipe_ridgeCV_zscore_confound_removal_wcategorical_averaged-source_Schaefer400x17_nodenoise_UKB_3353_z-beh_UKB_1893_subs_FC_all_cogs_Fluid_intelligence_score-2.0-rseed_123456-cv_res.csv
    beh_file = 'UKB_5000_subs_FC_all_cogs'
    pipe = 'ridgeCV_zscore_confound_removal_wcategorical'
    first = 'Fluid_intelligence_score-2.0'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}{beh_file}_Fluid_intelligence_score-2.0-rseed_123456-cv_res.csv')
    res['beh'] = first
    behs = pd.read_table('/data/project/ukb_reliability_in_prediction/prediction_UKB/code/opts/UKB_behs_all2.txt', header=None)
elif pipe == 'ridgeCV_z_averaged_behs':
    beh_file = 'UKB_1890_subs_FC_all_cogs_retest_average'
    pipe = 'ridgeCV_zscore'
    first = 'TMT_B_duration_to_complete-2.0'
    # First load empirical results, then append all simulation res to it
    # ridgeCV_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695-beh_interview_age_interview_age-rseed_123456-cv_res.csv
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}{beh_file}_TMT_B_duration_to_complete-2.0-rseed_123456-cv_res.csv')
    res['beh'] = first
    behs = pd.read_table('/data/project/ukb_reliability_in_prediction/prediction_UKB/code/opts/UKB_all_cogs_retest_023.txt', header=None)
elif pipe == 'ridgeCV_z_t2_behs':
    beh_file = 'UKB_1890_subs_FC_all_cogs'
    pipe = 'ridgeCV_zscore'
    first = 'TMT_B_duration_to_complete-2.0'
    # First load empirical results, then append all simulation res to it
    # ridgeCV_averaged-source_Schaefer400x17_WM+CSF+GS_hcpaging_695-beh_interview_age_interview_age-rseed_123456-cv_res.csv
    res = pd.read_csv(f'{in_path}/pipe_{pipe}{opts}{beh_file}_TMT_B_duration_to_complete-2.0-rseed_123456-cv_res.csv')
    res['beh'] = first
    behs = pd.read_table('/data/project/ukb_reliability_in_prediction/prediction_UKB/code/opts/UKB_all_cogs_retest_023.txt', header=None)


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
