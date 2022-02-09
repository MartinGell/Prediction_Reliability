# Imports
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler # for preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn import metrics
#from sklearn.pipeline import make_pipeline


# Set params
kern = 'linear'     # SVR kernel
C_param = 0.01      # C - taken from hyperparm tooning
tol_param = 0.001   # tolerance - same as above
k = 10              # CV n folds
n = 5               # CV n repeats
designator = 'test' # char designation of output file

# paths
#wd = Path('/home/mgell/Work/Prediction')
wd = Path('/data/project/impulsivity/prediction_HCPA')
out_dir = wd / 'res'

# load data and define leave out set
# table of subs (rows) by regions (columns)
path2FC = wd / 'text_files/FC_Nevena_Power2013_VOIs-combiSubs-rsFC-meanROI_GSR-5mm.csv'
FCs = pd.read_csv(path2FC)
FCs.pop('subs1')

# load behavioural measures
path2beh = wd / 'text_files/beh_Nevena_subs.csv'
tab = pd.read_csv(path2beh) # beh data
tab = tab.loc[:, ["Strength_Unadj"]]

# remove hold out data
X, X_val, y, y_val = train_test_split(FCs, tab, test_size=0.2, random_state=42)

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

# Save
out_file = out_dir / 'cv' / f"{designator}cv_res.csv"
cv_res.to_csv(out_file, index=False)

out_file = out_dir / 'validation' / f"{designator}validation_res.csv"
val_res.to_csv(out_file, index=False)

out_file = out_dir / 'predicted' / f"{designator}predicted_res.csv"
np.savetxt(out_file, y_pred, delimiter=',')