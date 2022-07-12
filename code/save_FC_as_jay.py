
# Load all files, unpickle them, and save them to a new file

import os
import numpy as np
import pandas as pd
import datatable as dt

# save into one table
nodes = 400
cols = ((nodes*nodes)-nodes)/2
data = np.empty((0,int(cols))) # data should be subs*regions
subs = np.empty((0,1))         # subs should be subs*1

# list files
in_path = os.getcwd()
files = os.listdir(in_path)

for f_i in files:
    f = np.load(f_i)
    fc = f['correlation_matrix']

    # extract vec
    FC = fc[np.triu_indices_from(fc,k=1)] # excludes diagonal

    # remove if NaN/None/missing/inf/etc
    if any(np.isnan(FC)):
        print(f'skipping {f_i} found {np.count_nonzero(np.isnan(FC).astype(int))} NaNs')
        continue
        #print('NOT REMOVNG')

    # zscore
    zFC = np.arctanh(FC)

    # save
    data = np.append(data, np.atleast_2d(zFC), axis=0)

    # save sub id
    name = f_i.split('_')
    subs = np.append(subs, np.atleast_2d(name[-1].split('.')[0]), axis=0)

# Save to a new file
print('Saving...')   
d = pd.concat([pd.DataFrame(subs, columns=['subID']), pd.DataFrame(data)], axis=1)
DT = dt.Frame(d)
DT.to_jay("/tmp/tmp_amir/for_Martin/Schaefer400x17_nodenoise_UKB_5000.jay")

#DT.to_jay("/data/project/impulsivity/prediction_simulations/input/test.jay")




# Load again to see diff in speed and check df looks the same after converting to pandas
# DT_new = dt.fread("/data/project/impulsivity/prediction_simulations/input/test.jay")

# jay_data = DT_new.to_pandas()
# jay_data.keys()
# jay_data.head()
