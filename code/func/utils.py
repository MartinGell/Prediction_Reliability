
# imports
import numpy as np
import pandas as pd
from scipy import stats



__all__ = ['filter_outliers','sort_files','transform2SD']

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