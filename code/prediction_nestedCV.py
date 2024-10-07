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

from func.utils import filter_outliers, sort_files, transform2SD, cor_true_pred_pearson, cor_true_pred_spearman, prep_confs
from func.models import model_choice
from sklearn.model_selection import ShuffleSplit, cross_validate, learning_curve, train_test_split, RepeatedKFold, KFold, GridSearchCV, GroupShuffleSplit, permutation_test_score


### Set params ###
FC_file = sys.argv[1]
beh_file = sys.argv[2]
beh = sys.argv[3]
pipe = sys.argv[4]


# SET UP
predict = True          # Predict? E.g. if only subsampling?
subsample = False       # Subsample data and compute learning curves?
remove_confounds = False # Remove confounds?
zscr = True             # zscore features

# Split validation before CV?
val_split = False       # Split data to train and held out validation?
val_split_size = 0.1    # Size of validation held out sample

# CV
k_inner = 5             # k folds for hyperparam search
k_outer = 10            # k folds for CV
n_outer = 5             # n repeats for CV
rs = 123456             # Random state: int for reproducibility or None

# NAMING OUTPUT
#res_folder = 'test'    # save results separately to ...
#designator = 'test'    # string designation of output file (begining)

# CONFOUNDS
load_confs = False   # False = confs in beh_file, otherwise it loads them from empirical data (see /func/utils.py)
# HCP A
#confounds = ['interview_age', 'gender']
#categorical = ['gender'] # of which categorical?
# HCP YA
#confounds = ['Age', 'Sex'] #['Age', 'Sex', 'FS_IntraCranial_Vol']
#categorical = ['Sex'] # of which categorical?
# UKB
#confounds = ['Age_when_attended_assessment_centre-2.0', 'sex']
#categorical = ['sex'] # of which categorical?
# ABCD
#confounds = ['interview_age', 'gender']
#categorical = ['gender'] # of which categorical?


# SAMPLING
if subsample:
    # subsample_Ns doesnt include test set. So if 10%, max can be full N - 10%
    #subsample_Ns = np.geomspace(200,580,4).astype('int') # HCP sampling
    subsample_Ns = np.geomspace(250,4450,7).astype('int') # UKB sampling, 50 less participants in case some are removed along the way 
    n_sample = 100      # number of samples to draw from data
    k_sample = 0.1      # fraction of data to use as test set
    res_folder = 'subsamples'
    print(f'\nSubsampling: {subsample_Ns}, each {n_sample} times with {k_sample*100}% left out\n')

score_pearson = metrics.make_scorer(cor_true_pred_pearson, greater_is_better=True)
score_spearman = metrics.make_scorer(cor_true_pred_spearman, greater_is_better=True)

scoring = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2", "r": score_pearson, "Rho": score_spearman}


#%%
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
print(f'\nUsing {beh}')
print(f'Behaviour data shape: {tab_all.shape}')

# attach confounds to tab_all if not there already before filtering
if remove_confounds:
    if load_confs:
        tab_all = prep_confs(tab_all, confounds, wd, beh_file)

# remove outliers
tab_all = filter_outliers(tab_all,beh)
print(f'Behaviour data shape: {tab_all.shape}') # just to check

# load data and define leave out set
# table of subs (rows) by regions (columns)
if FC_file == 'HCP2016FreeSurferSubcortical_abcd_baselineYear1Arm1_rest_3517.jay':
    print(f'\nUsing combined {FC_file} and HCP2016FreeSurferSubcortical_abcd_baselineYear1Arm1_rest_3435.jay')
    #
    path2FC_1 = wd / 'input' / FC_file
    path2FC_2 = wd / 'input' / 'HCP2016FreeSurferSubcortical_abcd_baselineYear1Arm1_rest_3435.jay'
    FCs_all_1 = dt.fread(path2FC_1)
    FCs_all_1 = FCs_all_1.to_pandas()
    FCs_all_2 = dt.fread(path2FC_2)
    FCs_all_2 = FCs_all_2.to_pandas()
    FCs_all = pd.concat([FCs_all_1, FCs_all_2], axis=0, ignore_index=True)
else:
    print(f'\nUsing {FC_file}')
    path2FC = wd / 'input' / FC_file
    #path2FC = Path(os.path.dirname(wd))
    #path2FC = path2FC / 'Preprocess_HCP' / 'res' / FC_file
    #FCs_all = pd.read_csv(path2FC)
    FCs_all = dt.fread(path2FC)
    #FCs_all.materialize(to_memory=True)
    FCs_all = FCs_all.to_pandas()

print(f'FC data shape: {FCs_all.shape}')

# Filter FC subs based on behaviour subs
tab, FCs = sort_files(tab_all, FCs_all)

# set up X and y for prediction
target = tab.loc[:, [beh]]
FCs.pop(FCs.keys()[0])
print('\nFCs after removing subjects:')
print(FCs.head())

# optionaly zscore FCs rowwise -> no data leakage
if zscr:
    print('\nZscoring FCs...')
    FCs = FCs.apply(lambda V: zscore(V), axis=1, result_type='broadcast')
    print('zscored')
else:
    print('\nNot zscoring!')

# set up for confound removal
if remove_confounds:
    print('\nSetting up for confound removal...')
    nested, model, grid = model_choice(pipe,X=FCs,confound=confounds,cat_columns=categorical)
    FCs.reset_index(inplace=True, drop=True)
    print(f'Running prediction with {model}')

    for conf_i in confounds:
        print(f'Adding {conf_i}')
        conf2remove = tab[f'{conf_i}']
        conf2remove.reset_index(inplace=True, drop=True)
        #FCs = FCs.append(tab[f'{conf_i}'], ignore_index=True)
        FCs[f'{conf_i}'] = conf2remove
else:
    print('\nNot removing confounds!')
    nested, model, grid = model_choice(pipe)
    print(f'Running prediction with {model}')

#%%
# remove hold out data
if val_split:
    X, X_val, y, y_val = train_test_split(FCs, target, test_size=val_split_size, random_state=rs)
else:
    X = FCs
    y = target

# CV set up
inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=rs)
outer_cv = RepeatedKFold(n_splits=k_outer, n_repeats=n_outer, random_state=rs)

# extra stuff for naming files
beh_f = beh_file.split('.')[0]
beh_f = beh_f.split('/')

# Only predict if requested
if predict:
    print('\nRunning Prediction...')
    # Run CV 
    if nested == 1: # Nested CV with parameter optimization
        print('Using nested CV!')
        print(f'Hyperparam search with {n_outer}x{k_outer}x{k_inner} over:')
        print(f'{grid}\n')
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1,
            cv=inner_cv, scoring="neg_root_mean_squared_error", verbose=3) # on Juseless n_jobs=None    
        scores = cross_validate(grid_search, X, np.ravel(y), scoring=scoring, cv=outer_cv,
            return_train_score=True, return_estimator=True, verbose=3, n_jobs=1)
        # results
        cv_res = pd.DataFrame(scores)
        for i in cv_res.loc[:,'estimator']: print(i.best_estimator_)             # put to utils?
    elif nested == 0: # non-nested CV
        print('Using vanilla CV!')
        print(f'CV with {n_outer}x{k_outer}\n')
        scores = cross_validate(model, X, np.ravel(y), scoring=scoring, cv=outer_cv,
            return_train_score=True, return_estimator=True, verbose=3, n_jobs=1)
        # results
        cv_res = pd.DataFrame(scores)
        if pipe == 'ridgeCV':                                                   # put to utils?
            for i in scores['estimator']: print(i.alpha_)
        elif pipe == 'ridgeCV_zscore':                                          # put to utils?
            for i in scores['estimator']: print(i[1].alpha_)                       # put to utils?
    elif nested == 99:
        print('Using stratified vanilla CV!')
        splits = (n_outer*k_outer)
        train_size = 0.7
        group = 'Family_ID'
        print(f'CV with {splits} splits of {round((1-train_size)*100)}% of groups out')
        print(f'Grouping variable: {group}\n')
        outer_cv = GroupShuffleSplit(n_splits=splits, train_size=train_size, random_state=rs)
        scores = cross_validate(model, X, np.ravel(y), groups=tab[group], scoring=scoring, cv=outer_cv,
            return_train_score=True, return_estimator=True, verbose=3, n_jobs=1)
        # results
        cv_res = pd.DataFrame(scores)
        if pipe == 'ridgeCV': # put to utils?
            for i in scores['estimator']: print(i.alpha_)
        elif pipe.__contains__('confound'):
            for i in scores['estimator']: print(i[2].alpha_)
        else:
            for i in scores['estimator']: print(i[1].alpha_)
    elif nested == 8: # permutation
        score_empirical, perm_scores, pval = permutation_test_score(model, X, np.ravel(y), scoring="r2", cv=outer_cv, n_permutations=100)
        print(f"Score on original data: {score_empirical:.2f} (p-value: {pval:.3f})")
         


    # Print results
    mean_accuracy = cv_res.mean()
    print(f'Overall MEAN accuracy:')
    print(mean_accuracy)

    sd_accuracy = cv_res.std()
    print(f'Overall SD accuracy:')
    print(sd_accuracy)

    # Save results of CV
    src_fc = ''.join(FC_file.split('.')[0:-1])
    if zscr:
        src_fc = f'{src_fc}_zscored'
    
    # CV results
    out_file = out_dir / 'cv'
    out_file.mkdir(parents=True, exist_ok=True)
    out_file = out_file / f"pipe_{pipe}-source_{src_fc}-beh_{beh_f[len(beh_f)-1]}_{beh}-rseed_{rs}-cv_res.csv"
    print(f'saving: {out_file}')
    cv_res.to_csv(out_file, index=False)

    # Averaged CV results
    out_file = out_dir / 'mean_accuracy'
    out_file.mkdir(parents=True, exist_ok=True)
    out_file = out_file / f"pipe_{pipe}_averaged-source_{src_fc}-beh_{beh_f[len(beh_f)-1]}_{beh}-rseed_{rs}-cv_res.csv"
    print(f'saving averaged accuracy: {out_file}')
    mean_accuracy.to_frame().transpose().to_csv(out_file, index=False)

    print('\nFINISHED WITH PREDICTION\n')

#%%
if val_split:
    print('\nWARNING: TAKING FIRST ESTIMATOR FROM ALL CVs.')
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
    print('\nComputing learning curve...')
    # cv
    outer_cv = ShuffleSplit(n_splits=n_sample, test_size=k_sample, random_state=rs)
    if nested == 1:
        print('Using nested CV!')
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1,
            cv=inner_cv, scoring="neg_root_mean_squared_error", verbose=3) # on Juseless n_jobs=None
        scores = learning_curve(grid_search, X, np.ravel(y), train_sizes=subsample_Ns, return_times=True, shuffle=True,
            cv=outer_cv, scoring='r2', verbose=3, n_jobs=1, random_state=rs)
    elif nested == 0:
        print('Using vanilla CV!')
        scores = learning_curve(model, X, np.ravel(y), train_sizes=subsample_Ns, return_times=True, shuffle=True,
            cv=outer_cv, scoring='r2', verbose=3, n_jobs=1, random_state=rs)

    # results
    train_size, train_scores, test_scores = scores[:3]
    sample_test_res = pd.DataFrame(np.transpose(test_scores), columns=train_size)
    sample_train_res = pd.DataFrame(np.transpose(train_scores), columns=train_size)

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

    print('\nFINISHED WITH LEARNING CURVES\n')
