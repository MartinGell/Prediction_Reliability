
# imports
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import Lasso, Ridge, ElasticNet, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler



__all__ = ['model_choice','filter_outliers','sort_files','transform2SD','cor_true_pred_pearson','cor_true_pred_pearson']

def model_choice(pipe):
    if pipe == 'svr':
        nested = 1 # using nested cv
        model = SVR()
        kernel = ["linear"]
        tolerance = [1e-3]
        C = [0.001, 0.01, 0.1, 1] # for age: [0.01, 0.1, 0.5, 1]
        grid = dict(kernel=kernel, tol=tolerance, C=C)
    elif pipe == 'svr_y_q':
        nested = 1 # using nested cv
        model = SVR()
        model = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(n_quantiles=400, output_distribution="normal"))
        kernel = ["linear"]
        tolerance = [1e-3]
        C = [0.001, 0.01, 0.1, 1] # for age: [0.01, 0.1, 0.5, 1]
        grid = {'regressor__kernel': kernel, 'regressor__tol': tolerance, 'regressor__C': C}
    elif pipe == 'lasso':
        nested = 1 # using nested cv
        model = Lasso(max_iter=1e4) #, selection='random')
        #tolerance = [1e-3]
        alphas = [1, 2, 5, 10, 100, 1e3, 1e4] #[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 5, 10, 100, 1e3, 1e4]
        grid = dict(alpha=alphas)#, tol=tolerance)
    elif pipe == 'ridge':
        nested = 1 # using nested cv
        model = Ridge()
        alphas = [100, 500, 1e3] #[1, 10, 100, 500, 1e3, 1e4] #[1e-4, 1e-3, 1e-2, 0.1, 1, 5, 10, 100, 1e3, 1e4]
        grid = dict(alpha=alphas)
    elif pipe == 'ridge_simple':
        nested = 0 # NOT using nested cv
        model = Ridge(alpha=1)
        grid = []
    elif pipe == 'ridgeCV':
        nested = 0 # using nested cv
        alphas = [100, 500, 1e3, 1e4] #[1, 10, 100, 500, 1e3, 1e4] #[1e-4, 1e-3, 1e-2, 0.1, 1, 5, 10, 100, 1e3, 1e4]        
        model = RidgeCV(alphas=alphas, store_cv_values=True, scoring="neg_root_mean_squared_error")
        grid = []
    elif pipe == 'ridge_learning_curve':
        nested = 1 # using nested cv
        model = Ridge()
        alphas = [100, 500, 1e3] #[1e-4, 1e-3, 1e-2, 0.1, 1, 5, 10, 100, 1e3, 1e4]
        grid = dict(alpha=alphas)
    elif pipe == 'kridge':
        nested = 1 # using nested cv
        model = KernelRidge()
        kernel = ["linear"]
        alphas = [1] #[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4]
        grid = dict(kernel=kernel, alpha=alphas)
    elif pipe == 'enet':
        nested = 1 # using nested cv
        model = ElasticNet(max_iter=1e4)
        alphas = [0.1, 1, 5] #1e-4, 1e-2, 1e-1, [1e-2,0.1]  
        l1_ratios = [0.2, 0.5, 0.7] #np.arange(0, 1, 0.15)
        grid = dict(alpha=alphas,l1_ratio=l1_ratios)
    else:
        raise Exception(f'Unknown model: {pipe}! Please use one of possible options')
    
    return nested, model, grid



def filter_outliers(tab, beh):
    tab['z'] = stats.zscore(tab.loc[:,[beh]])
    outliers = tab.loc[abs(tab['z']) > 3]
    tab = tab.loc[abs(tab['z']) < 3]
    if not outliers.empty:
        print(f'Removed {len(outliers)} outliers')
        print(f'{outliers}')
    
    return tab


def sort_files(tab, FCs):
    print('Sorting and filtering')
    FCs = FCs.sort_values(by=FCs.keys()[0])
    tab = tab.sort_values(by=tab.keys()[0])

    FCs = FCs[FCs.iloc[:,0].isin(tab.iloc[:,0])]
    tab = tab[tab.iloc[:,0].isin(FCs.iloc[:,0])]
    # FTR_TRGT = FTR.join(TRGT, how='inner') => same thing!
    if not all(FCs.iloc[:,0].to_numpy() == tab.iloc[:,0].to_numpy()):
        raise Exception('ERROR: Subjects are not the same between FCs and behaviour')
    
    print(f'NEW Behaviour data shape: {tab.shape}')
    print(f'NEW FC data shape: {FCs.shape}')

    return tab, FCs


def transform2SD(tab, beh, type):
    # call as tab, beh = transform2SD(tab, beh) # this way beh gets replaced by label
    if beh == 'nih_tlbx_agecsc_dominant':
        print('Splitting based on NIH norms: mean 100, sd 15')

        if type == 'split2bins':
            print(f'Using: {type}')
            label = f'{beh}_SD'
            tab[label] = 2
            tab[label] = np.where(tab[beh] >  100, tab[label] +1, tab[label])
            tab[label] = np.where(tab[beh] >= 115, tab[label] +1, tab[label])
            tab[label] = np.where(tab[beh] <= 100, tab[label] -1, tab[label])
            tab[label] = np.where(tab[beh] <=  85, tab[label] -1, tab[label])
            #plt.hist(tab[label],bins=5,histtype = 'bar',rwidth=0.8)
        elif type == 'outlier':
            print(f'Using: {type}')
            label = f'{beh}_SD'
            tab[label] = 1
            tab[label] = np.where(tab[beh] > 115, tab[label] +1, tab[label])
            tab[label] = np.where(tab[beh] <  85, tab[label] -1, tab[label])
    #label = f'{tab}_SD'
    #tab[label] = stats.zscore(tab.loc[:,[beh]])
    #outliers = tab.loc[abs(tab['z']) < 1]
    #tab = tab.loc[abs(tab['z']) > 1]


    print(beh)
    print(tab.shape)
    print(tab[beh].to_numpy())

    return tab, label


def cor_true_pred_pearson(y_true, y_pred):
    # >>> x = [1,2,3,4,5]
    # >>> y = [1,2,3,4,5]
    # >>> cor_true_pred_pearson(x,y)
    # 1.0
    # >>> y = [2,3,4,5,6]
    # >>> cor_true_pred_pearson(x,y)
    # 1.0
    # >>> y = [-1,-2,-3,-4,-5]
    # >>> cor_true_pred_pearson(x,y)
    # -1.0
    # >>> y = [1,2,3,4,10]
    # >>> cor_true_pred_pearson(x,y)
    # 0.8944271909999159
    # >>> y = [1,2,3,4,np.nan]
    # >>> cor_true_pred_pearson(x,y)
    # Traceback (most recent call last):
    # ... ValueError: array must not contain infs or NaNs
    cor, p = stats.pearsonr(y_true, y_pred)
    return cor

def cor_true_pred_spearman(y_true, y_pred):
    # >>> x = [1,2,3,4,5]
    # >>> y = [1,2,3,4,5]
    # >>> cor_true_pred_spearman(x,y)
    # 0.9999999999999999
    # >>> y = [2,3,4,5,6]
    # >>> cor_true_pred_spearman(x,y)
    # 0.9999999999999999
    # >>> y = [1,2,3,4,10]
    # >>> cor_true_pred_spearman(x,y)
    # 0.9999999999999999
    # >>> y = [-1,-2,-3,-4,-5]
    # >>> cor_true_pred_spearman(x,y)
    # -0.9999999999999999
    # >>> y = [-1,-2,-3,-4,np.nan]
    # >>> cor_true_pred_spearman(x,y)
    # nan
    # x = [1,2,3,4,5,6]
    # >>> y = [-1,-2,-3,4,5,6]
    # >>> cor_true_pred_spearman(x,y)
    # 0.7714285714285715
    # >>> y = [3,2,1,4,5,6]
    # >>> cor_true_pred_spearman(x,y)
    # 0.7714285714285715
    cor, p = stats.spearmanr(y_true, y_pred)
    return cor
