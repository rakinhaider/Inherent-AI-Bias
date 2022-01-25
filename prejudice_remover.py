"""
"""
import os
import tempfile
import pickle
import subprocess
import aif360
import numpy
import numpy as np
import pandas as pd
import importlib.util
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import fmin_cg


def _load_modules():
    # Loading pr
    k_path = aif360.__path__[0]
    pr_path = os.path.join(k_path, 'algorithms', 'inprocessing',
                           'kamfadm-2012ecmlpkdd', 'fadm/lr', 'pr.py')
    spec = importlib.util.spec_from_file_location('pr', pr_path)
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    return pr


pr = _load_modules()


class MyLRwPRType4(pr.LRwPRType4):
    N_S = 1

    def __init__(self, eta=1.0, C=1.0):
        super().__init__(eta, C)

    def fit(self, X, y, ns=N_S, itype=0, **kwargs):
        """ train this model

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            feature vectors of samples
        y : array, shape = (n_samples)
            target class of samples
        ns : int
            number of sensitive features. currently fixed to N_S
        itype : int
            type of initialization method
        kwargs : any
            arguments to optmizer
        """

        # rearrange input arguments
        s = np.atleast_1d(np.squeeze(np.array(X)[:, -ns]).astype(int))
        if self.fit_intercept:
            X = np.c_[np.atleast_2d(X)[:, :-ns], np.ones(X.shape[0])]
        else:
            X = np.atleast_2d(X)[:, :-ns]

        # check optimization parameters
        if not 'disp' in kwargs:
            kwargs['disp'] = False
        if not 'maxiter' in kwargs:
            kwargs['maxiter'] = 100

        # set instance variables
        self.n_s_ = ns
        self.n_sfv_ = 2
        self.c_s_ = np.array([np.sum(s == si).astype(np.float)
                              for si in range(self.n_sfv_)])
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]

        # optimization
        self.init_coef(itype, X, y, s)
        self.coef_ = fmin_cg(self.loss,
                             self.coef_,
                             fprime=self.grad_loss,
                             args=(X, y, s),
                             **kwargs)

        # get final loss
        self.f_loss_ = self.loss(self.coef_, X, y, s)


def fill_missing_with_mean(D, default=0.0):
    """ fill missing value with the means of non-missing values in the column

    Parameters
    ----------
    D : array, shape(n, m)
        raw data matrix
    default : float
        default value if all values are NaN

    Returns
    -------
    D : array, shape(n, m)
        a data matrix whose missing values are filled
    """

    for i in range(D.shape[1]):
        if np.any(np.isnan(D[:, i])):
            v = np.mean(D[np.isfinite(D[:, i]), i])
            if np.isnan(v):
                v = default
            D[np.isnan(D[:, i]), i] = v

    return D


class PrejudiceRemover(BaseEstimator, ClassifierMixin):
    """
    """
    def __init__(self, eta=1.0, C=1.0, sensitive_attr="", class_attr="",
                 favorable_label="", all_sensitive_attributes="",
                 privileged_value=""):
        """
        """
        self.eta = eta
        self.C = C
        self.sensitive_attr = sensitive_attr
        self.class_attr = class_attr
        self.favorable_label = favorable_label
        self.all_sensitive_attributes = all_sensitive_attributes
        self.privileged_value = privileged_value
        self.model = MyLRwPRType4(eta=self.eta, C=self.C)

    def _create_file_in_kamishima_format(self, df, class_attr,
                                         positive_class_val, sensitive_attrs,
                                         single_sensitive, privileged_vals):
        """Format the data for the Kamishima code and save it."""
        x = []
        for col in df:
            if col != class_attr and col not in sensitive_attrs:
                x.append(np.array(df[col].values, dtype=np.float64))
        x.append(np.array(single_sensitive.isin(privileged_vals),
                          dtype=np.float64))
        x.append(np.array(df[class_attr] == positive_class_val,
                          dtype=np.float64))

        fd, name = tempfile.mkstemp()
        os.close(fd)
        np.savetxt(name, np.array(x).T)
        return name

    def fit(self, X, y):
        """Learns randomized model with less bias

        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.

        Returns:
            self
        """
        if isinstance(y, numpy.ndarray):
            y = pd.Series(y, index=X.index, name=self.class_attr)
        if self.sensitive_attr == '':
            self.sensitive_attr = X.columns[-1]
        if self.class_attr == '':
            self.class_attr = y.name
        X, y = self._kamishima_format_xy(X, y)
        self.model.fit(X, y, 1, itype=2)
        self.classes_ = np.array([1-self.favorable_label, self.favorable_label])
        return self


    def predict(self, X):
        """Predict class labels for the given samples.
        Args:
            X (pandas.DataFrame): Test samples.
        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        y = pd.Series(np.ones(X.shape[0]), index=X.index, name=self.class_attr)
        X, y = self._kamishima_format_xy(X, y)
        return self.classes_[self.model.predict(X)]

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        y = pd.Series(np.ones(X.shape[0]), index=X.index, name=self.class_attr)
        X, y = self._kamishima_format_xy(X, y)
        return self.model.predict_proba(X)

    def _kamishima_format_xy(self, X, y):
        save_df = pd.concat([X, y], axis=1)
        sens_df = pd.Series(X[self.sensitive_attr], name=self.sensitive_attr)
        file_name = self._create_file_in_kamishima_format(save_df,
                      self.class_attr, self.favorable_label,
                      self.all_sensitive_attributes, sens_df,
                      np.array([self.privileged_value]))

        D = np.loadtxt(file_name)
        # split data and process missing values
        y = np.array(D[:, -1])
        X = fill_missing_with_mean(D[:, :-1])
        return X, y
