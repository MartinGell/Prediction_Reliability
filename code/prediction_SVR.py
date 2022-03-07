#%%
# Imports
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVR
#from sklearn.preprocessing import StandardScaler # for preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn import metrics
#from sklearn.pipeline import make_pipeline


### Set params ###
kern = 'linear'         # SVR kernel
C_param = 0.01          # C - taken from hyperparm tooning
tol_param = 0.001       # tolerance - same as above
k = 10                  # CV n folds
n = 5                   # CV n repeats
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

# load data and define leave out set
# table of subs (rows) by regions (columns)
path2FC = wd / 'text_files/FC_Nevena_Power2013_VOIs-combiSubs-rsFC-meanROI_GSR-5mm.csv'
FCs = pd.read_csv(path2FC)
#FCs.iloc[:,12:40]
#FCs['subs1'] = tab['Subject']

# Filter FC subs based on behaviour subs
FCs = FCs[FCs.iloc[:,0].isin(tab.iloc[:,0])]
tab = tab.loc[:, ["Strength_Unadj"]]
FCs.pop('subs1')

# remove hold out data
if val_split:
    X, X_val, y, y_val = train_test_split(FCs, tab, test_size=val_split_size, random_state=42)
else:
    X = FCs
    y = tab



### Prediction ###
# set model params
model = SVR(kernel=kern,tol=tol_param,C=C_param) # based on separate param tooning
# SVR params can be also set wiht set_params() but this seems less controlled
params = model.get_params

# Train and evaluate model with CV
scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]
cv_setup = RepeatedKFold(n_splits=k, n_repeats=n, random_state=42)
print("Running CV")
print(f"Using params: {params} and scoring: {scoring}")
# run CV
scores = cross_validate(model, X, np.ravel(y), scoring=scoring, cv=cv_setup,
return_train_score=False, return_estimator=True, verbose=3, n_jobs=1) # currently takes +-3.4mins on juseless?

cv_res = pd.DataFrame(scores)

# Save CV results
out_file = out_dir / 'cv' / f"{designator}cv_res.csv"
cv_res.to_csv(out_file, index=False)

# Save average CV results



if val_split:
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