import numpy
import numpy as np
import pandas
from sklearn.naive_bayes import _BaseNB, MultinomialNB, BernoulliNB
from inherent_bias.geomnb import GeomNB
from sklearn.utils import check_array


class MixedModelNB(_BaseNB):
    def __init__(self, *, class_prior=None, alpha=1e-9,
                 multinom_indices=[], geom_indices=[], bern_indices=[],
                 fit_prior=True):
        super(MixedModelNB, self).__init__()
        self.class_prior = class_prior
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.multinomial = MultinomialNB(alpha=alpha, fit_prior=fit_prior,
                                         class_prior=class_prior)
        self.geom = GeomNB(alpha=alpha, fit_prior=fit_prior,
                           class_prior=class_prior)
        self.bern = BernoulliNB(alpha=alpha, fit_prior=fit_prior,
                                class_prior=class_prior)
        self.multinom_indices = multinom_indices
        self.geom_indices = geom_indices
        self.bern_indices = bern_indices

    def _check_X(self, X):
        return check_array(X)

    def _split_x(self, X):
        mult_x = X[:, self.multinom_indices]
        geom_x = X[:, self.geom_indices]
        bern_x = X[:, self.bern_indices]
        return mult_x, geom_x, bern_x

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
        X = self._check_X(X)
        mult_x, geom_x, bern_x = self._split_x(X)
        trained_model = None
        if len(self.multinom_indices) > 0:
            self.multinomial.fit(mult_x, y)
            trained_model = self.multinomial
        if len(self.geom_indices) > 0:
            self.geom.fit(geom_x, y)
            trained_model = self.geom
        if len(self.bern_indices) > 0:
            self.bern.fit(bern_x, y)
            trained_model = self.bern
        self.classes_ = trained_model.classes_
        self.class_count_ = trained_model.class_count_
        if self.fit_prior:
            self.class_prior = trained_model.class_prior
            self.class_log_prior_ = trained_model.class_log_prior_
        else:
            self.class_log_prior_ = np.log(self.class_prior)
        return self

    def _joint_log_likelihood(self, X):
        X = self._check_X(X)
        mult_x, geom_x, bern_x = self._split_x(X)
        models = [(self.bern_indices, self.bern, bern_x),
                  (self.geom_indices, self.geom, geom_x),
                  (self.multinom_indices, self.multinomial, mult_x)]
        jll = np.zeros((X.shape[0], len(self.classes_)))
        models_used = 0
        for ind, mod, data in models:
            if len(ind) > 0:
                jll += mod._joint_log_likelihood(data)
                models_used += 1

        jll = jll - (models_used - 1) * self.class_log_prior_

        return jll