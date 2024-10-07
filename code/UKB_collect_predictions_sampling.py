
# Collect predictions
from glob import glob
from pathlib import Path
import sys
import pandas as pd
import numpy as np


# sample used
#beh = 'grip'
beh = sys.argv[1]
opts = '-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_'

# paths
wd = Path('/data/project/ukb_reliability_in_prediction/prediction_UKB/res') 
in_path = wd / 'subsamples_new' / 'learning_curve'
out_path = wd / 'learning_curve_collected'

# which beh was simulated? excluding rel values.
# e.g.: 'interview_age_wnoise'
if beh == 'age':
    beh_file = 'Age_when_attended_assessment_centre_wnoise'
    pipe = 'ridgeCV_zscore'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/test-pipe_ridgeCV_zscore-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_Age_when_attended_assessment_centre_Age_when_attended_assessment_centre-2.0-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    res_mean = pd.DataFrame(np.mean(res,0)).transpose()
    res_sd = pd.DataFrame(np.std(res,0)).transpose()
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'grip':
    beh_file = 'Hand_grip_strength_mean_lr_wnoise'
    pipe = 'ridgeCV_zscore'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/test-pipe_ridgeCV_zscore-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_Hand_grip_strength_mean_lr_Hand_grip_strength_mean_lr-2.0-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    res_mean = pd.DataFrame(np.mean(res,0)).transpose()
    res_sd = pd.DataFrame(np.std(res,0)).transpose()
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
elif beh == 'TMT':
    #test-pipe_ridgeCV_zscore-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_TMT_B_duration_to_complete_wnoise_rel_09_92_TMT_B_duration_to_complete-2.0_log_trans-rseed_123456-cv_res.csv
    beh_file = 'TMT_B_duration_to_complete_wnoise'
    pipe = 'ridgeCV_zscore'
    # First load empirical results, then append all simulation res to it
    res = pd.read_csv(f'{in_path}/test-pipe_ridgeCV_zscore-source_Schaefer400x17_nodenoise_UKB_5000_z-beh_TMT_B_duration_to_complete_TMT_B_duration_to_complete-2.0_log_trans-rseed_123456-cv_res.csv')
    res['reliability'] = 1.0
    res_mean = pd.DataFrame(np.mean(res,0)).transpose()
    res_sd = pd.DataFrame(np.std(res,0)).transpose()
    reliabilities = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]


# which files
f_designator = pipe+opts+beh_file
print(f'Looking for: {f_designator}_rel_*')

for rel_i in reliabilities:
    rel_i_str = str(rel_i)
    print(rel_i_str)
    files = glob(f"{in_path}/test-pipe_{f_designator}_rel_{rel_i_str.replace('.','')}_*")

    for f_i in files:
        # f = pd.read_csv(f_i)
        # f['reliability'] = rel_i
        # res_mean = res_mean.append(pd.DataFrame(np.mean(f,0)).transpose(), ignore_index=True)
        # res_sd = res_sd.append(pd.DataFrame(np.std(f,0)).transpose(), ignore_index=True)
        f = pd.read_csv(f_i)
        f['reliability'] = rel_i
        if not all(f.columns == res.columns):
            print(f.columns)
        new_cols = {x: y for x, y in zip(f.columns, res.columns)}
        res_mean = res_mean.append(pd.DataFrame(np.mean(f,0)).transpose().rename(columns=new_cols), ignore_index=True)
        res_sd = res_sd.append(pd.DataFrame(np.std(f,0)).transpose().rename(columns=new_cols), ignore_index=True)


# Save
out_path.mkdir(parents=True, exist_ok=True)
out_file = out_path / f'{pipe}{opts}{beh_file}_mean_all_new.csv'
res_mean.to_csv(out_file, index=False)

out_file = out_path / f'{pipe}{opts}{beh_file}_sd_all_new.csv'
res_sd.to_csv(out_file, index=False)