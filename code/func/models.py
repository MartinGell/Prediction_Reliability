
# imports
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, Ridge, ElasticNet, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR, LinearSVR

from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from func.heuristicModels import LinearSVRHeuristicC, SVRHeuristicC
from func.confound_removal import ConfoundRemover



__all__ = ['model_choice']

def model_choice(pipe, X = None, confound = None, cat_columns = None):
    # if X is None:
    #     X = pd.DataFrame()
    #     confound = pd.DataFrame()
    #     cat_columns = []

    if pipe == 'svr':
        nested = 1 # using nested cv
        model = SVR()
        kernel = ["linear"]
        #tolerance = [1e-3]
        C = [0.001, 0.01, 0.1] # for age: [0.01, 0.1, 0.5, 1]
        grid = dict(kernel=kernel, C=C)
    elif pipe == 'svr_heuristic':
        nested = 0 # using nested cv
        model = SVRHeuristicC(kernel="linear")
        grid = []
    elif pipe == 'svr_heuristic_zscore':
        nested = 0 # using nested cv
        model = make_pipeline(StandardScaler(),SVRHeuristicC(kernel="linear"))
        grid = []
    elif pipe == 'svr_heuristic_zscore_confound_removal_wcategorical':
        nested = 0 # using nested cv
        categorical_columns = cat_columns#['Gender']
        continous_columns = [col for col in X.columns.to_list() if col not in cat_columns]
        preprocessor = ColumnTransformer(
            transformers=[
                ("cont",StandardScaler(), continous_columns),
                ("cat", "passthrough", categorical_columns)],
        # remainder=PassThrough() -> will drop any further columns by default
        )
        model = make_pipeline(
            preprocessor,
            ConfoundRemover(n_confounds=len(confound)),
            RidgeCV(alphas=alphas, store_cv_values=True, scoring="neg_root_mean_squared_error")
            )
        grid = []
    elif pipe == 'svr_L1':
        nested = 1 # using nested cv
        model = LinearSVR(loss='epsilon_insensitive')
        C = [0.001, 0.1, 1, 100] # for age: [0.01, 0.1, 0.5, 1]
        grid = dict(C=C)
    elif pipe == 'svr_L1_heuristic':
        nested = 0 # using nested cv
        model = LinearSVRHeuristicC(loss='epsilon_insensitive')
        grid = []
    elif pipe == 'svr_L1_heuristic_zscore':
        nested = 0 # using nested cv
        model = make_pipeline(StandardScaler(),LinearSVRHeuristicC(loss='epsilon_insensitive'))
        grid = []
    elif pipe == 'svr_L2':
        nested = 1 # using nested cv
        model = LinearSVR(loss='squared_epsilon_insensitive')
        C = [0.001, 0.1, 1, 100] # for age: [0.01, 0.1, 0.5, 1]
        grid = dict(C=C)
    elif pipe == 'svr_L2_heuristic':
        nested = 0 # using nested cv
        model = LinearSVRHeuristicC(loss='squared_epsilon_insensitive')
        grid = []
    elif pipe == 'svr_L2_heuristic_zscore':
        nested = 0 # using nested cv
        model = make_pipeline(StandardScaler(),LinearSVRHeuristicC(loss='squared_epsilon_insensitive'))
        grid = []
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
        alphas = [1e3, 1e4] #[1, 10, 100, 500, 1e3, 1e4] #[1e-4, 1e-3, 1e-2, 0.1, 1, 5, 10, 100, 1e3, 1e4]
        grid = dict(alpha=alphas)
    elif pipe == 'ridgeCV':
        nested = 0 # using nested cv
        alphas = [1, 10, 100, 1e3, 1e4, 1e5] #[1, 10, 100, 500, 1e3, 1e4] #[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4]        
        model = RidgeCV(alphas=alphas, store_cv_values=True, scoring="neg_root_mean_squared_error")
        grid = []
    elif pipe == 'ridgeCV_zscore':
        nested = 0 # using nested cv
        alphas = [1, 10, 100, 1e3, 1e4, 1e5] #[1, 10, 100, 500, 1e3, 1e4] #[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4]        
        model = make_pipeline(
            StandardScaler(),RidgeCV(alphas=alphas, store_cv_values=True, scoring="neg_root_mean_squared_error")
            )
        grid = []
    elif pipe == 'ridgeCV_zscore_stratified_KFold':
        nested = 99 # using nested cv
        alphas = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5] #[1, 10, 100, 500, 1e3, 1e4] #[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4]        
        model = make_pipeline(
            StandardScaler(),RidgeCV(alphas=alphas, store_cv_values=True, scoring="neg_root_mean_squared_error")
            )
        grid = []
    elif pipe == 'ridgeCV_zscore_confound_removal':
        nested = 0 # using nested cv
        alphas = [1, 10, 100, 1e3, 1e4, 1e5] #[1, 10, 100, 500, 1e3, 1e4] #[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4]        
        model = make_pipeline(
            StandardScaler(), 
            ConfoundRemover(n_confounds=len(confound)), 
            RidgeCV(alphas=alphas, store_cv_values=True, scoring="neg_root_mean_squared_error")
            )
        grid = []
    elif pipe == 'ridgeCV_zscore_confound_removal_wcategorical':
        nested = 0 # using nested cv
        alphas = [1, 10, 100, 1e3, 1e4, 1e5] #[1, 10, 100, 500, 1e3, 1e4] #[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4]
        categorical_columns = cat_columns#['Gender']
        continous_columns = [col for col in X.columns.to_list() if col not in cat_columns]
        preprocessor = ColumnTransformer(
            transformers=[
                ("cont",StandardScaler(), continous_columns),
                ("cat", "passthrough", categorical_columns)],
        # remainder=PassThrough() -> will drop any further columns by default
        )
        model = make_pipeline(
            preprocessor,
            ConfoundRemover(n_confounds=len(confound)),
            RidgeCV(alphas=alphas, store_cv_values=True, scoring="neg_root_mean_squared_error")
            )
        grid = []
    elif pipe == 'BAD': # Example how not to set up confound regression with categorical variables
        nested = 99 # using nested cv
        alphas = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5] #[1, 10, 100, 500, 1e3, 1e4] #[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4]        
        categorical_columns = cat_columns#['Gender']
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", "passthrough", categorical_columns)],
                remainder=StandardScaler())
        model = make_pipeline(
            preprocessor, 
            ConfoundRemover(n_confounds=len(confound)), 
            RidgeCV(alphas=alphas, store_cv_values=True, scoring="neg_root_mean_squared_error")
            )
        grid = []
    elif pipe == 'ridgeCV_zscore_stratified_KFold_confound_removal_wcategorical':
        nested = 99 # using nested cv
        alphas = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5] #[1, 10, 100, 500, 1e3, 1e4] #[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4]        
        categorical_columns = cat_columns#['Gender']
        continous_columns = [col for col in X.columns.to_list() if col not in cat_columns]
        preprocessor = ColumnTransformer(
            transformers=[
                ("cont",StandardScaler(), continous_columns),
                ("cat", "passthrough", categorical_columns)],
        # remainder=PassThrough() -> will drop any further columns by default
        )
        model = make_pipeline(
            preprocessor,
            ConfoundRemover(n_confounds=len(confound)),
            RidgeCV(alphas=alphas, store_cv_values=True, scoring="neg_root_mean_squared_error")
            )
        grid = []
    elif pipe == 'kridge':
        nested = 1 # using nested cv
        model = KernelRidge()
        kernel = ["linear"]
        alphas = [1e-3, 1e-2, 0.1, 1, 10, 100, 1e3]
        grid = dict(kernel=kernel, alpha=alphas)
    elif pipe == 'kridge_zscore': #note this combinations can produce odd results
        nested = 1 # using nested cv
        model = make_pipeline(StandardScaler(),KernelRidge())
        kernel = ["linear"]
        alphas = [1e-3, 1e-2, 0.1, 1, 10, 100, 1e3]
        grid = dict(kernelridge__kernel=kernel, kernelridge__alpha=alphas)
    elif pipe == 'enet':
        nested = 1 # using nested cv
        model = ElasticNet(max_iter=1e4)
        alphas = [0.1, 1, 5] #1e-4, 1e-2, 1e-1, [1e-2,0.1]  
        l1_ratios = [0.2, 0.5, 0.7] #np.arange(0, 1, 0.15)
        grid = dict(alpha=alphas,l1_ratio=l1_ratios)
    else:
        raise Exception(f'Unknown model: {pipe}! Please use one of possible options')
    
    return nested, model, grid

