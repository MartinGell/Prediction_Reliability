# Load all files, unpickle them, and save them to a new file

from pathlib import Path
import numpy as np
import pandas as pd

# paths
path2beh = Path('PATH_2_UKB_5100_subs_FC_behav.csv')

# Load beh_file
beh = pd.read_csv(path2beh / 'UKB_5100_subs_FC_behav.csv')
subs = beh['eid']

# some extra stuff
nodes = 400
cols = ((nodes*nodes)-nodes)/2

#for f_i in files:
for sub in subs:
    f_i = f'FC_{int(sub)}.npz'
    f = np.load(f_i)
    fc = f['correlation_matrix']

    # extract vec
    FC = fc[np.triu_indices_from(fc,k=1)] # excludes diagonal

    # remove if NaN/None/missing/inf/etc
    if any(np.isnan(FC)):
        print(f'skipping {f_i} found {np.count_nonzero(np.isnan(FC).astype(int))} NaNs')
        continue

    # Check if each FC has enough connections
    if np.count_nonzero(FC) < cols:
        print(f'skipping {f_i} FC different length: {np.count_nonzero(FC)} connections')
        continue

