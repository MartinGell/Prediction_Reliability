
# imports
from sklearn.svm import LinearSVR, SVR
from func.utils import heuristic_C


# Define classes that use heruristics for hyperparameters


# SVR + C heuristic
# inherit class from SVR
class LinearSVRHeuristicC(LinearSVR):

    # Overwrite fit method to use heuristic C as HP
    def fit(self, X, y, sample_weight=None):

        # calculate heuristic C
        C = heuristic_C(X)
        # TODO Give back a dictionary from heuristic functions

        # set C value
        self.C = C

        # call super fit method
        super().fit(X, y, sample_weight=sample_weight)
    
        return  self


class SVRHeuristicC(SVR):

    # Overwrite fit method to use heuristic C as HP
    def fit(self, X, y, sample_weight=None):

        # calculate heuristic C
        C = heuristic_C(X)
        # TODO Give back a dictionary from heuristic functions

        # set C value
        self.C = C

        # call super fit method
        super().fit(X, y, sample_weight=sample_weight)
    
        return  self

