#%%
# Imports
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV, cross_val_score


### Set params ###
model = SVR()
kernel = ["linear"]
tolerance = [1e-3]
C = [0.001, 0.01, 0.1, 1, 10, 100]
grid = dict(kernel=kernel, tol=tolerance, C=C)
scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]

k_inner = 5             # k folds for hyperparam search
k_outer = 10            # k folds for CV
n_outer = 1             # n repeats for CV
rs = 1                  # random state: int or None

designator = 'test'     # char designation of output file
val_split = False       # Split data to train and held out validation?
val_split_size = 0.2    # Size of validation held out sample

#%%

# paths
wd = Path('/home/mgell/Work/Prediction_HCP')
#wd = Path('/data/project/impulsivity/prediction_HCPA')
out_dir = wd / 'res'

# load behavioural measures
path2beh = wd / 'text_files/beh_Nevena_subs.csv'
tab = pd.read_csv(path2beh) # beh data
#y = tab[N:N,N]
#y_val = tab[N:N,N]

# load data and define leave out set
# table of subs (rows) by regions (columns)
path2FC = wd / 'text_files/FC_Nevena_Power2013_VOIs-combiSubs-rsFC-meanROI_GSR-5mm.csv'
FCs = pd.read_csv(path2FC)
#FCs = FCs.iloc[0:339,:]
#FCs = FCs.iloc[:,-1]

# Filter FC subs based on behaviour subs
#FCs = FCs[FCs.iloc[:,0].isin(tab.iloc[:,0])]
tab = tab.loc[:, ["Strength_Unadj"]] #Strength_Unadj nih_tlbx_agecsc_dominant
FCs.pop('subs1')

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
cv_res.iloc[0,2].best_params_



#%%
# explore results of crossval
cv_res = search_res.cv_results_
cv_res = pd.DataFrame(cv_res)
#best_params = cv_res.loc[cv_res['rank_test_score'] == 1,['params']]
print(f'Best hyperparams for SVR based on {k} folds and {n} repeats:')
print(search_res.best_params_)

# extract the best model and evaluate it
print("[INFO] evaluating...")
bestModel = search_res.best_estimator_
print("R2: {:.2f}".format(bestModel.score(X_val, y_val)))

Y_pred = bestModel.predict(X_val)
np.corrcoef(Y_pred,np.ravel(y_val))

# Plot results


# Save
np.savetxt(r'/home/mgell/Work/FC/brainsmash/NAT_surrogate_maps.csv', cv_res, delimiter=',')
