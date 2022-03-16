
# imports
import numpy as np
import pandas as pd
from scipy import stats
from logging import exception



__all__ = ['filter_outliers','sort_files']

def filter_outliers(tab, beh):
    tab['z'] = stats.zscore(tab.loc[:,[beh]])
    tab = tab.loc[abs(tab['z']) < 3]
    outliers = tab.loc[abs(tab["z"]) > 3]
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

    if not all(FCs.iloc[:,0].to_numpy() == tab.iloc[:,0].to_numpy()):
        raise Exception('ERROR: Subjects are not the same between FCs and behaviour')

    return tab, FCs
