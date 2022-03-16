#%%
# Imports
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from logging import exception
from scipy import stats
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, KFold, GridSearchCV, cross_val_score
#import matplotlib.pyplot as plt



### Set params ###
#FC_file = 'FC_Nevena_Power2013_VOIs-combiSubs-rsFC-meanROI_GSR-5mm.csv'
FC_file = 'zFC_seitzman_nodes_rfMRI_REST1_AP-subs_664-params_FC_gm_FSL025_no_overlap_dt_flt0.1_0.01.csv'
#FC_file = sys.argv[1]
#beh_file = 'text_files/beh_Nevena_subs.csv'
beh_file = 'beh_HCP_A_motor.csv'
#beh = "Strength_Unadj"
beh = 'nih_tlbx_agecsc_dominant'

model = SVR()
kernel = ["linear"]
tolerance = [1e-3]
C = [0.0001, 0.001, 0.01, 0.1, 1]
grid = dict(kernel=kernel, tol=tolerance, C=C)
scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"] #need to create a new scorer for r

k_inner = 5             # k folds for hyperparam search
k_outer = 10            # k folds for CV
n_outer = 5             # n repeats for CV
rs = None               # random state: int or None

designator = 'test'     # char designation of output file
val_split = False       # Split data to train and held out validation?
val_split_size = 0.2    # Size of validation held out sample

#%%

# paths
#wd = Path('/home/mgell/Work/Prediction_HCP')
wd = Path('/data/project/impulsivity/Prediction_HCP')
out_dir = wd / 'res'

# load behavioural measures
path2beh = wd / 'text_files' / beh_file
tab = pd.read_csv(path2beh) # beh data
tab = tab.dropna(subset = [beh]) # drop nans if there are in beh of interest
print(f'Using {beh}')
print(f'Behaviour data shape: {tab.shape}')

# remove outliers -> put into utils
tab['z'] = stats.zscore(tab.loc[:,[beh]])
tab = tab.loc[abs(tab['z']) < 3]
outliers = tab.loc[abs(tab["z"]) > 3]
if not outliers.empty:
    print(f'Removed {len(outliers)} outliers')
    print(f'{outliers}')

# load data and define leave out set
# table of subs (rows) by regions (columns)
path2FC = Path(os.path.dirname(wd))
path2FC = path2FC / 'Preprocess_HCP' / 'res' / FC_file
FCs = pd.read_csv(path2FC)
print(f'Using {FC_file}')
print(f'FC data shape: {FCs.shape}')

# Filter FC subs based on behaviour subs -> put into utils
print('Sorting and filtering')
FCs = FCs.sort_values(by=FCs.keys()[0])
tab = tab.sort_values(by=tab.keys()[0])

FCs = FCs[FCs.iloc[:,0].isin(tab.iloc[:,0])]
tab = tab[tab.iloc[:,0].isin(FCs.iloc[:,0])]

if not all(FCs.iloc[:,0].to_numpy() == tab.iloc[:,0].to_numpy()):
    raise Exception('ERROR: Subjects are not the same between FCs and behaviour')

tab = tab.loc[:, [beh]]
FCs.pop(FCs.keys()[0])

#%%
# remove hold out data
if val_split:
    X, X_val, y, y_val = train_test_split(FCs, tab, test_size=val_split_size, random_state=rs)
else:
    X = FCs
    y = tab

# CV set up
inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=rs)
outer_cv = RepeatedKFold(n_splits=k_outer, n_repeats=n_outer, random_state=rs)

# Nested CV with parameter optimization
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
	cv=inner_cv, scoring="neg_root_mean_squared_error", verbose=3) # on Juseless n_jobs=None
scores = cross_validate(grid_search, X, np.ravel(y), scoring=scoring, cv=outer_cv,
    return_train_score=True, return_estimator=True, verbose=3, n_jobs=1)


cv_res = pd.DataFrame(scores)
print(f'Best hyperparams from nested CV {n_outer}x{k_outer}x{k_inner}:')
#print(cv_res.iloc[0,2].best_params_)
for i in cv_res.loc[:,'estimator']: print(i.best_estimator_)

print(f'Overall accuracy: {cv_res.mean()}')

#%%
# Save cv results
out_file = out_dir / 'cv'
out_file.mkdir(parents=True, exist_ok=True)
out_file = out_file / f"{''.join(FC_file.split('.')[0:-1])}_cv_res.csv"
print(f'saving: {out_file}')
cv_res.to_csv(out_file, index=False)

print('FINISHED')

#%%
if val_split:
    print('WARNING: TAKING FIRST ESTIMATOR FROM ALL CVs.')
    print('IF DIFFERENT WILL IMPACT FOLLOWING RESULTS.')
    params = cv_res.iloc[0,2].get_params

    # Predict on validation data
    # Now fit entire training set and predict leave out set
    print(f"Fitting model with all training data: {params}")
    model.fit(X, np.ravel(y))
    # predict on validation data
    print("evaluating on left out validation data")
    y_pred = model.predict(X_val)

    # validation results to be saved
    val_r = np.corrcoef(y_pred,np.ravel(y_val))
    val_r2 = model.score(X_val,y_val)
    val_MAE = metrics.mean_absolute_error(y_val,y_pred)
    val_rMSE = metrics.mean_squared_error(y_val,y_pred)
    val_res = {
        "r":[val_r[0,1]],
        "r2":[val_r2],
        "MAE":[val_MAE],
        "rMSE":[val_rMSE**(1/2)]
    }
    val_res = pd.DataFrame(val_res)
    print(f"on validation r(predicted,observed) = {val_r}")

    # Save validation results
    out_file = out_dir / 'validation' / f"{designator}validation_res.csv"
    val_res.to_csv(out_file, index=False)

    out_file = out_dir / 'predicted' / f"{designator}predicted_res.csv"
    np.savetxt(out_file, y_pred, delimiter=',')
