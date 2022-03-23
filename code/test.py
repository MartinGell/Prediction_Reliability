from sklearn.preprocessing import QuantileTransformer, quantile_transform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.utils.fixes import parse_version

# `normed` is being deprecated in favor of `density` in histograms
if parse_version(matplotlib.__version__) >= parse_version("2.1"):
    density_param = {"density": True}
else:
    density_param = {"normed": True}


#tab = tab.loc[:, [beh]]
#y = tab
y_trans = quantile_transform(y, n_quantiles=400, output_distribution="normal", copy=True).squeeze()

f, (ax0, ax1) = plt.subplots(1, 2)

ax0.hist(y, bins=50, **density_param)
ax0.set_ylabel("Probability")
ax0.set_xlabel("Target")
#ax0.text(s="Target distribution", x=1.2e5, y=9.8e-6, fontsize=12)
#ax0.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

ax1.hist(y_trans, bins=50, **density_param)
ax1.set_ylabel("Probability")
ax1.set_xlabel("Target")
#ax1.text(s="Transformed target distribution", x=-6.8, y=0.479, fontsize=12)

f.suptitle("interview age", y=0.04)
#f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])


grid = {'regressor__kernel': kernel, 'regressor__tol': tol, 'regressor__C': C}