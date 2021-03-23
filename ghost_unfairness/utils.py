from aif360.metrics import (
    BinaryLabelDatasetMetric,
    ClassificationMetric
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from ghost_unfairness.fair_dataset import FairDataset, default_mappings
from copy import deepcopy


def get_single_prot_default_map():
    metadata = default_mappings.copy()
    metadata['protected_attribute_maps'] = [{1.0: 'Male', 0.0: 'Female'}]
    return metadata


def get_dataset_metrics(fd_train,
                        verbose=False):
    unprivileged_groups = fd_train.unprivileged_groups
    privileged_groups = fd_train.privileged_groups
    metrics = BinaryLabelDatasetMetric(fd_train,
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

    if verbose:
        print('Mean Difference:', metrics.mean_difference())
        print('Dataset Base Rate', metrics.base_rate(privileged=None))
        print('Privileged Base Rate', metrics.base_rate(privileged=True))
        print('Protected Base Rate', metrics.base_rate(privileged=False))
        print('Disparate Impact:', metrics.disparate_impact())
        # print('Confusion Matrix:', metrics.binary_confusion_matrix())

    return metrics.mean_difference(), metrics.disparate_impact()


def get_positive_rate(cmetrics, privileged=None, positive=True):
    if positive:
        # Compute Delta_A^+
        return cmetrics.true_positive_rate(privileged)
    else:
        # Compute Delta_A^-
        return cmetrics.false_positive_rate(privileged)


def get_classifier_metrics(clf, data,
                           verbose=False,
                           pos_rate=False):
    unprivileged_groups = data.unprivileged_groups
    privileged_groups = data.privileged_groups

    data_pred = data.copy()
    data_x, data_y = data.get_xy(keep_protected=False)
    data_pred.labels = clf.predict(data_x)

    metrics = ClassificationMetric(data,
                                   data_pred,
                                   privileged_groups=privileged_groups,
                                   unprivileged_groups=unprivileged_groups)

    if verbose:
        print('Mean Difference:', abs(0 - metrics.mean_difference()))
        print('Disparate Impact:', abs(1 - metrics.disparate_impact()))
        # print('Confusion Matrix:', metrics.binary_confusion_matrix())
        print('Accuracy:', metrics.accuracy())

    m = [metrics.mean_difference(),
         metrics.disparate_impact(),
         metrics.accuracy(),
         metrics.accuracy(privileged=True),
         metrics.accuracy(privileged=False)]

    if pos_rate:
        for pg in [True, False]:
            for pos_rate in [True, False]:
                m.append(get_positive_rate(metrics, pg, pos_rate))
    return m


def plot_lr_boundary(clf, plt, label):
    # Retrieve the model parameters.
    b = clf.intercept_[0]
    w1, w2 = clf.coef_.T
    # Calculate the intercept and gradient of the decision boundary.
    c = -b / w2
    m = -w1 / w2

    # Plot the data and the classification with the decision boundary.
    xmin, xmax = -10, 20
    ymin, ymax = -10, 25
    xd = np.array([xmin, xmax])
    yd = m * xd + c
    plt.plot(xd, yd, lw=1, label=label)
    if label:
        plt.legend()


def get_model_properties(model):
    if isinstance(model, DecisionTreeClassifier):
        return model.get_depth()
    elif isinstance(model, LogisticRegression):
        return model.coef_
    elif isinstance(model, GaussianNB):
        return model.theta_, np.sqrt(model.sigma_)


def get_datasets(n_samples, n_features, n_redlin, kwargs,
                 train_random_state=47, test_random_state=23):
    temp_kwargs = deepcopy(kwargs)
    train_fd = FairDataset(n_samples, n_features, n_redlin,
                           **temp_kwargs, random_state=train_random_state)
    temp_kwargs = deepcopy(kwargs)
    test_fd = FairDataset(n_samples // 2, n_features, n_redlin,
                          **temp_kwargs, random_state=test_random_state)
    return train_fd, test_fd


def train_model(model_type, data, params):
    x, y = data.get_xy(keep_protected=False)

    model = model_type()
    # params[variant] = val
    model.set_params(**params)

    model = model.fit(x, y)

    return model


def get_groupwise_performance(train_fd, test_fd, model_type,
                              privileged=None,
                              params=None,
                              pos_rate=False):
    if privileged:
        train_fd = train_fd.get_privileged_group()

    elif privileged == False:
        train_fd = train_fd.get_unprivileged_group()

    if not params:
        params = get_model_params(model_type)

    model = train_model(model_type, train_fd, params)
    results = get_classifier_metrics(model, test_fd,
                                     verbose=False,
                                     pos_rate=pos_rate)

    return model, results


def get_model_params(model_type):
    if model_type == DecisionTreeClassifier:
        params = {'criterion': 'entropy',
                  'max_depth': 5,
                  'random_state': 47}
    elif model_type == LogisticRegression:
        params = {'class_weight': 'balanced',
                  'solver': 'liblinear'}
    elif model_type == GaussianNB:
        params = {}
    return params
