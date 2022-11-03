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
from sklearn.model_selection import ShuffleSplit, cross_validate, learning_curve, train_test_split, RepeatedKFold, KFold, GridSearchCV


### Set params ###
FC_file = sys.argv[1]
beh_file = sys.argv[2]
beh = sys.argv[3]
pipe = sys.argv[4]

k_inner = 5             # k folds for hyperparam search
k_outer = 10            # k folds for CV
n_outer = 5             # n repeats for CV
rs = 123456             # random state: int for reproducibility or None

predict = False          # predict or just subsample?
subsample = False       # Subsample data and compute learning curves?

remove_confounds = False # Remove confounds?
confs_in_file = True    # False = confs in beh file, otherwise it loads them from empirical data
confounds = ['interview_age', 'gender'] #['Age', 'Sex', 'FS_IntraCranial_Vol'] # 'FS_Total_GM_Vol'
categorical = ['gender'] #['Sex']   # of which categorical?

external_validation = True

zscr = True             # zscore features
val_split = False       # Split data to train and held out validation?
val_split_size = 0.2    # Size of validation held out sample

res_folder = 'external_validation'    # save results separately to ...
#designator = 'test'    # string designation of output file

if subsample:
    #subsample_Ns = np.array([195,295,395]) # these are only train + 55 test makes 250, 350 and 450
    subsample_Ns = np.geomspace(250,4450,7).astype('int')
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
    if confs_in_file:
        tab_all = prep_confs(tab_all, wd, FC_file)

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
#FCs_all.materialize(to_memory=True)
FCs_all = FCs_all.to_pandas()
print(f'\nUsing {FC_file}')
print(f'FC data shape: {FCs_all.shape}')

# Filter FC subs based on behaviour subs
tab, FCs = sort_files(tab_all, FCs_all)
# transform scores to SD_scores -> not sure if this is the right place for it
#tab, beh = transform2SD(tab, beh, 'outlier')

# set up X and y for prediction
target = tab.loc[:, [beh]]
FCs.pop(FCs.keys()[0])
print('\nFCs after removing subjects:')
print(FCs.head())

# optionaly zscore FCs rowwise -> no data leakage
src_fc = ''.join(FC_file.split('.')[0:-1])
if zscr:
    print('\nZscoring FCs...')
    FCs = FCs.apply(lambda V: zscore(V), axis=1, result_type='broadcast')
    print('zscored')
    src_fc = f'{src_fc}_zscored'
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


#FCs[tab_all.isna().any(axis=1)]

#%%
# remove hold out data
if val_split:
    print(f'\nSplitting into train and validation sets uisng {val_split_size} as validation...')
    X, X_val, y, y_val = train_test_split(FCs, target, test_size=val_split_size, random_state=rs)
    print(f'train size: {X.shape}')
    print(f'test size:  {X_val.shape}')
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
        print(f'{grid}')
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1,
            cv=inner_cv, scoring="neg_root_mean_squared_error", verbose=3) # on Juseless n_jobs=None    
        scores = cross_validate(grid_search, X, np.ravel(y), scoring=scoring, cv=outer_cv,
            return_train_score=True, return_estimator=True, verbose=3, n_jobs=1)
        # results
        cv_res = pd.DataFrame(scores)
        for i in cv_res.loc[:,'estimator']: print(i.best_estimator_)             # put to utils?
    elif nested == 0: # non-nested CV
        print('Using vanilla CV!')
        print(f'CV with {n_outer}x{k_outer}:')
        scores = cross_validate(model, X, np.ravel(y), scoring=scoring, cv=outer_cv,
            return_train_score=True, return_estimator=True, verbose=3, n_jobs=1)
        # results
        cv_res = pd.DataFrame(scores)
        if pipe == 'ridgeCV':                                                   # put to utils?
            for i in scores['estimator']: print(i.alpha_)
        elif pipe == 'ridgeCV_zscore':                                          # put to utils?
            for i in scores['estimator']: print(i[1].alpha_)                       # put to utils?


    # Print results
    mean_accuracy = cv_res.mean()
    print(f'Overall MEAN accuracy:')
    print(mean_accuracy)

    sd_accuracy = cv_res.std()
    print(f'Overall SD accuracy:')
    print(sd_accuracy)

    ## SAVE
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
    #print('\nWARNING: TAKING FIRST ESTIMATOR FROM ALL CVs.')
    #print('IF DIFFERENT WILL IMPACT FOLLOWING RESULTS.')
    #params = cv_res.iloc[0,2].get_params

    # Predict on validation data
    # Now fit entire training set and predict leave out set
    #print(f"Fitting model with all training data: {params}")
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
        "R2":[val_r2],
        "MAE":[val_MAE],
        "RMSE":[val_rMSE**(1/2)]
    }
    val_res = pd.DataFrame(val_res)
    print(f"on validation r(predicted,observed) = {val_r2}")

    # Save validation results
    out_file = out_dir / 'validation'
    out_file.mkdir(parents=True, exist_ok=True)
    out_file = out_file / f"pipe_{pipe}_averaged-source_{src_fc}-beh_{beh_f[len(beh_f)-1]}_{beh}-rseed_{rs}-validation_res.csv"
    print(f'saving averaged accuracy: {out_file}')
    val_res.to_csv(out_file, index=False)

    out_file = out_dir / 'predicted'
    out_file.mkdir(parents=True, exist_ok=True)
    out_file = out_file / f"pipe_{pipe}_averaged-source_{src_fc}-beh_{beh_f[len(beh_f)-1]}_{beh}-rseed_{rs}predicted_res.csv"
    print(f'saving averaged accuracy: {out_file}')
    np.savetxt(out_file, y_pred, delimiter=',')

if external_validation:
    #val_FC_file = 'Schaefer400x17_WM+CSF+GS_hcpya_771.jay'
    #val_beh_file = 'HCP_YA_beh_confs_all.csv'
    val_FC_file = 'Schaefer400x17_WM+CSF+GS_hcpya_316.jay'
    val_beh_file = 'HCP_YA_beh_confs_unrelated.csv'
    val_beh = 'CogCrystalComp_AgeAdj'
    #val_beh = 'Strength_AgeAdj'

    print(f'\nRunning extrenal validation on {val_FC_file} dataset')

    path2beh = wd / 'text_files' / val_beh_file
    val_tab_all = pd.read_csv(path2beh) # beh data
    val_tab_all = val_tab_all.dropna(subset = [val_beh]) # drop nans if there are in beh of interest
    print(f'\nUsing: {val_beh}')
    print(f'Behaviour data shape: {val_tab_all.shape}')

    # attach confounds to tab_all if not there already before filtering
    #NEED TO CHECK IF THIS WILL WORK WITH CONF REMOVAL
    #if remove_confounds:
    #    if confs_in_file:
    #        tab_all = prep_confs(tab_all, wd, val_FC_file)

    # remove outliers
    val_tab_all = filter_outliers(val_tab_all,val_beh)
    print(f'Behaviour data shape: {val_tab_all.shape}') # just to check

    # load data and define leave out set
    # table of subs (rows) by regions (columns)
    path2FC = wd / 'input' / val_FC_file
    #path2FC = Path(os.path.dirname(wd))
    #path2FC = path2FC / 'Preprocess_HCP' / 'res' / FC_file
    #FCs_all = pd.read_csv(path2FC)
    val_FCs_all = dt.fread(path2FC)
    #FCs_all.materialize(to_memory=True)
    val_FCs_all = val_FCs_all.to_pandas()
    print(f'\nUsing {val_FC_file}')
    print(f'FC data shape: {val_FCs_all.shape}')

    # Filter FC subs based on behaviour subs
    val_tab, val_FCs = sort_files(val_tab_all, val_FCs_all)
    # transform scores to SD_scores -> not sure if this is the right place for it
    #tab, beh = transform2SD(tab, beh, 'outlier')

    # set up X and y for prediction
    val_target = val_tab.loc[:, [val_beh]]
    val_FCs.pop(val_FCs.keys()[0])
    print('\nFCs after removing subjects:')
    print(val_FCs.head())

    if zscr:
        print('\nZscoring FCs...')
        FCs = FCs.apply(lambda V: zscore(V), axis=1, result_type='broadcast')
        print('zscored')
    else:
        print('\nNot zscoring!')

    ## PREDCTION
    print(f'\nTraining on full {FC_file} dataset...')
    model.fit(X, np.ravel(y))

    print(f'\nPredicting on external validation {val_FC_file} dataset...')
    val_target_pred = model.predict(val_FCs)

    # validation results to be saved
    val_r = np.corrcoef(val_target_pred,np.ravel(val_target))
    val_r2 = model.score(val_FCs,val_target)
    val_MAE = metrics.mean_absolute_error(val_target,val_target_pred)
    val_rMSE = metrics.mean_squared_error(val_target,val_target_pred)
    val_res = {
        "r":[val_r[0,1]],
        "R2":[val_r2],
        "MAE":[val_MAE],
        "RMSE":[val_rMSE**(1/2)]
    }
    val_res = pd.DataFrame(val_res)

    print(f'Accuracy on external validation:')
    print(val_res)

    # Save external validation results
    out_file = out_dir / 'external_validation'
    out_file.mkdir(parents=True, exist_ok=True)
    out_file = out_file / f"pipe_{pipe}_averaged-source_{src_fc}-beh_{beh_f[len(beh_f)-1]}_{beh}-rseed_{rs}-validation_in_{val_FC_file}_{val_beh}-res.csv"
    print(f'saving averaged accuracy: {out_file}')
    val_res.to_csv(out_file, index=False)


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
