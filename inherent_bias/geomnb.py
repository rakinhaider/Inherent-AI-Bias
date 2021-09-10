from sklearn.naive_bayes import _BaseDiscreteNB
from sklearn.utils.validation import column_or_1d
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import LabelBinarizer


class GeomNB(_BaseDiscreteNB):
    def __init__(self, *, class_prior=None, fit_prior=True, alpha=1e-9):
        super(GeomNB, self).__init__()
        self.class_prior = class_prior
        self.fit_prior = fit_prior
        self.alpha = alpha

    def _joint_log_likelihood(self, X):
        jll = 1 - self.feature_prob_ + 10e-23
        jll = np.log(jll)
        jll = np.matmul(jll, X.T)
        sum_log_prob = np.log(self.feature_prob_).sum(axis=1).reshape(-1, 1)
        jll += sum_log_prob
        jll += np.log(self.class_prior).reshape(-1, 1)
        return jll.T

    def fit(self, X, y, sample_weight=None):
        """Fit Geometric Naive Bayes according to X, y

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        X, y = self._validate_data(X, y)
        y = column_or_1d(y, warn=True)
        n_features = X.shape[1]

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_

        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        n_classes = len(labelbin.classes_)
        self._init_counters(n_classes, n_features)
        _, self.class_count_ = np.unique(y, return_counts=True)
        self.feature_means_ = safe_sparse_dot(Y.T, X)
        # Compute the average from sum.
        self.feature_means_ = np.divide(self.feature_means_,
                                        self.class_count_.reshape(-1, 1))
        self.feature_prob_ = 1 / (self.feature_means_ + self.alpha)
        if self.fit_prior or len(self.class_prior) == 0:
            self.class_prior = self.class_count_/np.sum(self.class_count_)

        # TODO: Add smoothing later.
        self.class_log_prior_ = np.log(self.class_prior)
        return self

    def _init_counters(self, n_effective_classes, n_features):
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)