import warnings
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import _safe_indexing


class ConfoundRemover(BaseEstimator, TransformerMixin):
    def __init__(self, model_confound=None, threshold=None,
                 n_confounds=0,
                 drop_confounds=True, n_jobs=None, verbose=0):
        """Transformer to remove n_confounds from the features.
        Subtracts the predicted features given confounds from features.
        Resulting residuals can be thresholded in case residuals are so small
        that rounding error can be informative.

        Parameters
        ----------
        model_confound : obj
            Scikit-learn compatible model used to predict all features
            independently using the confounds as features.
            The predictions of these models
            are then subtracted from each feature, defaults to
            LinearRegression().
        threshold : float | None
            All residual values after confound removal which fall under the
            threshold will be set to 0. None (default) means that no threshold
            will be applied.
        n_confounds : int
            Number of confounds inside of X.
            The last n_confounds columns in X will be used as confounds.
        n_jobs : int
            Number of jobs in parallel.
            See.: https://scikit-learn.org/stable/computing/parallelism.html
        verbose: int
            How verbose the output of Parallel Processes should be.
        """
        if model_confound is None:
            model_confound = LinearRegression()
        self.model_confound = model_confound
        self.threshold = threshold
        self.n_confounds = n_confounds
        self.drop_confounds = drop_confounds
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None, apply_to=None):
        """Fit confound remover

        Parameters
        ----------
        X : pandas.DataFrame | np.ndarray
            Training data. Includes features and n_confounds as the last
            n columns.
        y : pandas.Series | np.ndarray
            Target values.
        apply_to : array-like of int, slice, array-like of bool
            apply_to will be used to index the features inside of X
            (excluding the confound). The selected features will be confound
            removed. The keep as they are.

        """
        check_array(X)
        self.apply_to_ = apply_to
        if self.n_confounds <= 0:
            warnings.warn(
                'Number of confounds is 0 or below '
                'confound removal will not have any effect')
            return self
        # confounds = self.safe_select(X, slice(-self.n_confounds, None))
        confounds = _safe_indexing(X, slice(-self.n_confounds, None), axis=1)

        def fit_confound_models(t_X):
            _model = clone(self.model_confound)
            _model.fit(confounds, t_X)
            return _model

        # t_X = safe_select(X, slice(None, -self.n_confounds))
        t_X = _safe_indexing(X, slice(None, -self.n_confounds), axis=1)
        if self.apply_to_ is not None:
            t_X = _safe_indexing(t_X, self.apply_to_, axis=1)

        # self.models_confound_ = Parallel(
        #     n_jobs=self.n_jobs, verbose=self.verbose,
        #     **_joblib_parallel_args())(
        #     delayed(fit_confound_models)(_safe_indexing(X, i_X))
        #     for i_X in range(t_X.shape[1])
        # )

        self.models_confound_ = [
            fit_confound_models(_safe_indexing(X, i_X, axis=1))
            for i_X in range(t_X.shape[1])
        ]
        return self

    def transform(self, X):
        """Removes confounds from X.

        Parameters
        ----------
        X : pandas.DataFrame | np.ndarray
            Training data. Includes features and n_confounds as the last
            n columns.

        Returns
        -------
        out : np.ndarray
            Deconfounded X.
        """
        check_is_fitted(self)
        check_array(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.n_confounds <= 0:
            return X

        confounds = _safe_indexing(X, slice(-self.n_confounds, None), axis=1)
        X = _safe_indexing(X, slice(None, -self.n_confounds), axis=1)
        X = X.copy()
        idx = np.arange(0, X.shape[1])
        if self.apply_to_ is not None:
            idx = idx[self.apply_to_]
        for i_model, model in enumerate(self.models_confound_):
            t_idx = idx[i_model]
            t_pred = model.predict(confounds)
            X_res = X[:, t_idx] - t_pred
            if self.threshold is not None:
                X_res[np.abs(X_res) < self.threshold] = 0
            X[:, t_idx] = X_res

        if not self.drop_confounds:
            X = np.c_[X, confounds]
        return X

    def will_drop_confounds(self):
        return self.drop_confounds
