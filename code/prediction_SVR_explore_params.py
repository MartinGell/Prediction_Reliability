
# Imports
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV


# Set params
model = SVR()
k = 10 # n folds
n = 5  # n repeats
kernel = ["linear"]
tolerance = [1e-3, 1e-4, 1e-5, 1e-6]
C = [0.001, 0.01, 0.1, 1, 10, 100]
grid = dict(kernel=kernel, tol=tolerance, C=C)

# paths
#wd = Path('/home/mgell/Work/Prediction')
wd = Path('/data/project/impulsivity/prediction_HCPA')
out_dir = wd / 'res'

# load data and define leave out set
# table of subs (rows) by regions (columns)
path2FC = wd / 'text_files/FC_Nevena_Power2013_VOIs-combiSubs-rsFC-meanROI_GSR-5mm.csv'
FCs = pd.read_csv(path2FC)
#FCs = FCs.iloc[0:339,:]
#FCs = FCs.iloc[:,-1]
FCs.pop('subs1')

# load behavioural measures
path2beh = wd / 'text_files/beh_Nevena_subs.csv'
tab = pd.read_csv(path2beh) # beh data
#y = tab[N:N,N]
#y_val = tab[N:N,N]
tab = tab.loc[:, ["Strength_Unadj"]]


# remove hold out data
X, X_val, y, y_val = train_test_split(FCs, tab, test_size=0.2, random_state=0)

# Hyperparameter tooning
cv_setup = RepeatedKFold(n_splits=k, n_repeats=n, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
	cv=cv_setup, scoring="neg_root_mean_squared_error",verbose=2) # on Juseless n_jobs=None
print("[INFO] grid searching over the hyperparameters...")
search_res = grid_search.fit(X, np.ravel(y))

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