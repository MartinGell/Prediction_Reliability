
# Imports
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold, train_test_split

# Set params


# load data and define leave out set
# table of subs (rows) by regions (columns)
FCs = pd.read_csv('/home/mgell/Work/Prediction/text_files/FC_Nevena_Power2013_VOIs-combiSubs-rsFC-meanROI_GSR-5mm.csv')
#FCs = FCs.iloc[0:339,:]
#FCs = FCs.iloc[:,-1]
FCs.pop('subs1')

# cycle through measures with dif. reliability loop

# load behavioural measures
tab = pd.read_csv('/home/mgell/Work/Prediction/text_files/beh_Nevena_subs.csv') # beh data
tab = tab.loc[:, ['Strength_Unadj']]
#y = tab[N:N,N]
#y_val = tab[N:N,N]

# remove hold out data
X, X_val, y, y_val = train_test_split(FCs, tab, test_size=0.1, random_state=0)


# set up cross validation
cv = RepeatedKFold(n_splits=10, n_repeats=1)
model = ElasticNetCV(cv = cv)

# learn
model.fit(X,y)
print(model.alpha_)     # actually lambda
print(model.l1_ratio_)  # actually alpha
#print(model.best_score_) #???

# predict on validation data
xxxx = model.predict(X_val)
np.corrcoef(xxxx,np.ravel(y_val))