import os
import sys
from matplotlib import pyplot
import numpy as np
import pandas as pd

from pathlib import Path
from numpy import asarray, exp
from sklearn.neighbors import KernelDensity

beh_file = 'beh_HCP_A_motor.csv'
beh = 'interview_age'

wd = os.getcwd()
wd = Path(os.path.dirname(wd))
out_dir = wd / 'res'

# load behavioural measures
path2beh = wd / 'text_files' / beh_file
tab = pd.read_csv(path2beh) # beh data

d = tab.loc[:, [beh]].to_numpy()

d_mean = np.mean(d)
d_sd = np.std(d)


model = KernelDensity(bandwidth=50, kernel='gaussian')
#d = d.reshape((len(d),1))
model.fit(d)

values = asarray([value for value in range(int((min(d)-d_sd)),int((max(d)+d_sd)))])
values = values.reshape((len(values),1))
probab = model.score_samples(values)
probab = exp(probab)

pyplot.hist(d, bins=50, density=True)
pyplot.plot(values[:], probab)
pyplot.show