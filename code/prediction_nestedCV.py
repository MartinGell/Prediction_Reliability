#%%
# Imports
import os
import sys
import numpy as np
import pandas as pd
import datatable as dt

from pathlib import Path
from sklearn import metrics
from scipy.stats import zscore

from func.utils import filter_outliers, sort_files, transform2SD, cor_true_pred_pearson, cor_true_pred_spearman
from func.models import model_choice
from sklearn.model_selection import ShuffleSplit, cross_validate, learning_curve, train_test_split, RepeatedKFold, KFold, GridSearchCV
#import matplotlib.pyplot as plt


### Set params ###
pipe = sys.argv[4]
#FC_file = 'seitzman_nodes_average_runs_REST1_REST1_REST2_REST2-subs_651-params_FC_gm_FSL025_no_overlap_dt_flt0.1_0.01.csv'
FC_file = sys.argv[1]
#beh_file = 'beh_HCP_A_motor.csv'
beh_file = sys.argv[2]
#beh = 'interview_age' #'nih_tlbx_agecsc_dominant' #'interview_age'  'nih_tlbx_agecsc_dominant'
beh = sys.argv[3]

k_inner = 5             # k folds for hyperparam search
k_outer = 10            # k folds for CV
n_outer = 5             # n repeats for CV
rs = 123456             # random state: int for reproducibility or None

predict = True          # predict or just subsample?
subsample = True        # Subsample data and compute learning curves?

zscr = True             # zscore data
designator = 'test'     # char designation of output file
val_split = False       # Split data to train and held out validation?
val_split_size = 0.2    # Size of validation held out sample

#res_folder = 'exact_distribution'

if subsample:
    #subsample_Ns = np.array([195,295,395]) # these are only train + 55 test makes 250, 350 and 450
    subsample_Ns = np.geomspace(250,4500,8)/4500 # Fractions of total in case some rows from FC are removed
    n_sample = 100
    k_sample = 0.1
    res_folder = 'subsamples'
    print(f'Subsampling: {subsample}, each {n_sample} times with {k_sample*100}% left out')

score_pearson = metrics.make_scorer(cor_true_pred_pearson, greater_is_better=True)
score_spearman = metrics.make_scorer(cor_true_pred_spearman, greater_is_better=True)

scoring = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2", "r": score_pearson, "Rho": score_spearman}


#%%
# start message
nested, model, grid = model_choice(pipe)
print(f'Running prediction with {model}')

# paths
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
out_dir = wd / 'res' 
if 'res_folder' in locals():
    out_dir = out_dir / res_folder

# load behavioural measures
path2beh = wd / 'text_files' / beh_file
tab_all = pd.read_csv(path2beh) # beh data
tab_all = tab_all.dropna(subset = [beh]) # drop nans if there are in beh of interest
print(f'Using {beh}')
print(f'Behaviour data shape: {tab_all.shape}')

# remove outliers
tab_all = filter_outliers(tab_all,beh)
print(f'Behaviour data shape: {tab_all.shape}') # just to check

# load data and define leave out set
# table of subs (rows) by regions (columns)
path2FC = wd / 'input' / FC_file
#path2FC = Path(os.path.dirname(wd))
#path2FC = path2FC / 'Preprocess_HCP' / 'res' / FC_file
#FCs_all = pd.read_csv(path2FC)
FCs_all = dt.fread(path2FC)
FCs_all = FCs_all.to_pandas()
print(f'Using {FC_file}')
print(f'FC data shape: {FCs_all.shape}')

# Filter FC subs based on behaviour subs
tab, FCs = sort_files(tab_all, FCs_all)
# transform scores to SD_scores -> not sure if this is the right place for it
#tab, beh = transform2SD(tab, beh, 'outlier')
tab = tab.loc[:, [beh]]
FCs.pop(FCs.keys()[0])
print('FCs after removing subjects:')
print(FCs.head())

# optionaly zscore FCs
if zscr:
    FCs = FCs.apply(lambda V: zscore(V), axis=1, result_type='broadcast')
    print('FCs zscored')


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


# Only predict if requested
if predict:
    print('Running Prediction...')
    # Run CV 
    if nested == 1: # Nested CV with parameter optimization
        print('Using nested CV..')
        print(f'Hyperparam search with {n_outer}x{k_outer}x{k_inner} over:')
        print(f'{grid}')
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1,
            cv=inner_cv, scoring="neg_root_mean_squared_error", verbose=3) # on Juseless n_jobs=None    
        scores = cross_validate(grid_search, X, np.ravel(y), scoring=scoring, cv=outer_cv,
            return_train_score=True, return_estimator=True, verbose=3, n_jobs=1)
        # results
        cv_res = pd.DataFrame(scores)
        for i in cv_res.loc[:,'estimator']: print(i.best_estimator_)             # put to utils?
    elif nested == 0: # non-nested CV
        print('Using vanilla CV..')
        print(f'CV with {n_outer}x{k_outer}:')
        scores = cross_validate(model, X, np.ravel(y), scoring=scoring, cv=outer_cv,
            return_train_score=True, return_estimator=True, verbose=3, n_jobs=1)
        # results
        cv_res = pd.DataFrame(scores)
        if pipe == 'ridgeCV':                                                   # put to utils?
            for i in scores['estimator']: print(i.alpha_)
        elif pipe == 'ridgeCV_zscore':                                          # put to utils?
            for i in scores['estimator']: print(i[1].alpha_)                       # put to utils?
    elif nested == 99:
        print('Using a heuristic to determine hyperparams')
        print(f'CV with {n_outer}x{k_outer}:')
        #databased_C = heuristic_C(data_df=None)

    mean_accuracy = cv_res.mean()
    print(f'Overall MEAN accuracy:')
    print(mean_accuracy)

    sd_accuracy = cv_res.std()
    print(f'Overall SD accuracy:')
    print(sd_accuracy)

#%%
# Save cv results
beh_f = beh_file.split('.')[0]
beh_f = beh_f.split('/')

src_fc = ''.join(FC_file.split('.')[0:-1])
if zscr:
    src_fc = f'{src_fc}_zscored'

out_file = out_dir / 'cv'
out_file.mkdir(parents=True, exist_ok=True)
out_file = out_file / f"pipe_{pipe}-source_{src_fc}-beh_{beh_f[len(beh_f)-1]}_{beh}-rseed_{rs}-cv_res.csv"
print(f'saving: {out_file}')
cv_res.to_csv(out_file, index=False)

out_file = out_dir / 'mean_accuracy'
out_file.mkdir(parents=True, exist_ok=True)
out_file = out_file / f"pipe_{pipe}_averaged-source_{src_fc}-beh_{beh_f[len(beh_f)-1]}_{beh}-rseed_{rs}-cv_res.csv"
print(f'saving averaged accuracy: {out_file}')
mean_accuracy.to_frame().transpose().to_csv(out_file, index=False)

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

if subsample:
    print('Computing learning curve..')
    # cv
    outer_cv = ShuffleSplit(n_splits=n_sample, test_size=k_sample, random_state=rs)
    if nested == 1:
        print('Using nested CV..')
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1,
            cv=inner_cv, scoring="neg_root_mean_squared_error", verbose=3) # on Juseless n_jobs=None
        scores = learning_curve(grid_search, X, np.ravel(y), train_sizes=subsample_Ns, return_times=True, shuffle=True,
            cv=outer_cv, scoring='r2', verbose=3, n_jobs=1, random_state=rs)
    elif nested == 0:
        print('Using vanilla CV..')
        scores = learning_curve(model, X, np.ravel(y), train_sizes=subsample_Ns, return_times=True, shuffle=True,
            cv=outer_cv, scoring='r2', verbose=3, n_jobs=1, random_state=rs)

    # results
    train_size, train_scores, test_scores = scores[:3]
    #train_errors, test_errors = np.transpose(-train_scores), np.transpose(-test_scores)
    train_errors = np.transpose(train_scores)
    test_errors = np.transpose(test_scores)
    cols = list()
    #for i in train_size[:]: cols.append(f'x{str(i)}')
    for i in train_size[:]: cols.append(i+55)

    sample_test_res = pd.DataFrame(test_errors, columns=cols)
    sample_train_res = pd.DataFrame(train_errors, columns=cols)

    print(f'Overall MEAN accuracy:')
    print(sample_test_res.mean())

    print(f'Overall SD accuracy:')
    print(sample_test_res.std())
    
    # save train
    out_file = out_dir / 'learning_curve'
    out_file.mkdir(parents=True, exist_ok=True)
    out_file = out_file / f"train-pipe_{pipe}-source_{''.join(FC_file.split('.')[0:-1])}-beh_{beh_f[len(beh_f)-1]}_{beh}-rseed_{rs}-cv_res.csv"
    print(f'saving: {out_file}')
    sample_train_res.to_csv(out_file, index=False)

    # save test
    out_file = out_dir / 'learning_curve'
    out_file = out_file / f"test-pipe_{pipe}-source_{''.join(FC_file.split('.')[0:-1])}-beh_{beh_f[len(beh_f)-1]}_{beh}-rseed_{rs}-cv_res.csv"
    print(f'saving: {out_file}')
    sample_test_res.to_csv(out_file, index=False)

    #cv_res = pd.DataFrame(scores)
    #print(f'Best hyperparams from nested CV {n_outer}x{k_outer}x{k_inner}:')
    #for i in cv_res.loc[:,'estimator']: print(i.best_estimator_)
