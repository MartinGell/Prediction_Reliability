
# Load all files, unpickle them, and save them to a new file

import numpy as np
import pandas as pd
import datatable as dt

# paths
path2beh = '/tmp/tmp_amir/for_Martin'
outpath = '/data/project/ukb_reliability_in_prediction/input'

# Load beh_file
#beh = pd.read_csv(f'{path2beh}/UKB_5000_subs_FC_behav.csv')
x = np.linspace(0,99,100)
x = np.delete(x,41)
beh = pd.DataFrame(x, columns=['eid'])
sampled_subs = beh['eid']

# save into one table
nodes = 400
cols = ((nodes*nodes)-nodes)/2
data = np.empty((0,int(cols))) # data should be subs*regions
subs = np.empty((0,1))         # subs should be subs*1

for sub in sampled_subs:
    f_i = f'FC_{int(sub)}.npz'
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
    #zFC = np.arctanh(FC) #dont do this, its done later

    # save
    data = np.append(data, np.atleast_2d(FC), axis=0)

    # save sub id
    name = f_i.split('_')
    subs = np.append(subs, np.atleast_2d(name[-1].split('.')[0]), axis=0)

# Save to a new file
d = pd.concat([pd.DataFrame(subs, columns=['subID']), pd.DataFrame(data)], axis=1)

# Save as .jay
print('Saving...')
out = f'{outpath}/Schaefer400x17_nodenoise_UKB_5000.jay'

DT = dt.Frame(d)
DT.to_jay(out)

#DT.to_jay("/data/project/impulsivity/prediction_simulations/input/test.jay")




# Load again to see diff in speed and check df looks the same after converting to pandas
# DT_new = dt.fread("/data/project/impulsivity/prediction_simulations/input/test.jay")

# jay_data = DT_new.to_pandas()
# jay_data.keys()
# jay_data.head()
