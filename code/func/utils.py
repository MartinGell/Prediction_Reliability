
# imports
import numpy as np
import pandas as pd
from scipy import stats


__all__ = ['heuristic_C','filter_outliers','sort_files','transform2SD','cor_true_pred_pearson','cor_true_pred_pearson','test_cor_true_pred','prep_confs']


def heuristic_C(data_df=None):
    """
    Function from Vera/Kaustubh

    Calculate the heuristic C for linearSVR (Joachims 2002).

    Returns
    -------
    C : float
        Theoretically calculated hyperparameter C for a linear SVM.
    """

    if data_df is None:
        raise Exception('No data was provided.')

    C = 1/np.mean(np.sqrt((data_df**2).sum(axis=1)))
    print(f'Using C = {C} based on heuritic (C = 1/mean(sqrt(rowSums(data^2))))')
    # Formular Kaustubh: C = 1/mean(sqrt(rowSums(data^2)))

    return C


def filter_outliers(tab, beh, SD=3):
    print(f'\nFiltering outliers using {SD} SD...')
    tab['z'] = stats.zscore(tab.loc[:,[beh]])
    outliers = tab.loc[abs(tab['z']) > SD]
    tab = tab.loc[abs(tab['z']) < SD]
    if not outliers.empty:
        print(f'Removed {len(outliers)} outliers!')
        print(f'{outliers}')
    else:
        print('No outliers found!')
    
    return tab


def sort_files(tab, FCs):
    print('\nSorting and filtering...')
    #tab.iloc[:,0] = tab.iloc[:,0].astype(int)
    #FCs.iloc[:,0] = FCs.iloc[:,0].astype(int)

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
    # e.g. call as tab, beh = transform2SD(tab, beh, 'outlier') # this way beh gets replaced by label
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


def test_cor_true_pred(y_true, y_pred):
    r, p = stats.spearmanr(y_true, y_pred)
    rho, p = stats.pearsonr(y_true, y_pred)
    cor = {'rho': rho, 'r': r}
    return cor


def prep_confs(tab_all, confounds, wd, beh_file):
    if 'HCP_A_total' in beh_file:
        empirical_file = 'HCP_A_total.csv'
        path2file = wd / 'text_files' / 'rel' / empirical_file
        empirical_data = pd.read_csv(path2file)
    if 'HCP_A_cryst' in beh_file:
        empirical_file = 'HCP_A_cryst.csv'
        path2file = wd / 'text_files' / 'rel' / empirical_file
        empirical_data = pd.read_csv(path2file)
    if 'HCP_A_motor' in beh_file:
        empirical_file = 'HCP_A_motor.csv'
        path2file = wd / 'text_files' / 'rel' / empirical_file
        empirical_data = pd.read_csv(path2file)

    print(f'Getting confs from: {empirical_file}')
    for conf_i in confounds:
        print(f'Adding {conf_i}')
        conf2add = empirical_data[f'{conf_i}']
        conf2add.reset_index(inplace=True, drop=True)
        #FCs = FCs.append(tab[f'{conf_i}'], ignore_index=True)
        tab_all[f'{conf_i}'] = conf2add
    
    return tab_all

