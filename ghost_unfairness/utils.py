from aif360.metrics import (
    BinaryLabelDatasetMetric,
    ClassificationMetric
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from ghost_unfairness.fair_dataset import FairDataset, default_mappings
from ghost_unfairness.ds_fair_dataset import DSFairDataset
from copy import deepcopy
from scipy.special import erf
from math import sqrt
import scipy.stats as stats
import math
import matplotlib.pyplot as plt

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
        return cmetrics.true_positive_rate(privileged)
    else:
        return cmetrics.false_positive_rate(privileged)


def get_classifier_metrics(clf, data,
                           verbose=False,
                           sel_rate=False):
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
         abs(1 - metrics.disparate_impact()),
         metrics.accuracy(),
         metrics.accuracy(privileged=True),
         metrics.accuracy(privileged=False)]

    if sel_rate:
        for pg in [True, False]:
            for sr in [True, False]:
                m.append(get_positive_rate(metrics, pg, sr))

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
    if kwargs['ds']:
        temp_kwargs = deepcopy(kwargs)
        del temp_kwargs['ds']
        train_fd = DSFairDataset(
            n_samples, n_features, n_redlin, **temp_kwargs,
            random_state=train_random_state
        )
        temp_kwargs = deepcopy(kwargs)
        del temp_kwargs['ds']
        test_fd = DSFairDataset(n_samples // 2, n_features, n_redlin,
                                **temp_kwargs, random_state=test_random_state)
    else:     
        temp_kwargs = deepcopy(kwargs)
        del temp_kwargs['ds']
        train_fd = FairDataset(n_samples, n_features, n_redlin,
                           **temp_kwargs, random_state=train_random_state)
        temp_kwargs = deepcopy(kwargs)
        del temp_kwargs['ds']
        test_fd = FairDataset(n_samples // 2, n_features, n_redlin,
                              **temp_kwargs, random_state=test_random_state)
    return train_fd, test_fd


def train_model(model_type, data, params):
    x, y = data.get_xy(keep_protected=False)

    model = model_type(**params)
    # params[variant] = val
    # model.set_params(**params)

    model = model.fit(x, y)

    return model


def get_groupwise_performance(train_fd, test_fd, model_type,
                              privileged=None,
                              params=None,
                              pos_rate=False,
                              privileged_group=None,
                              unprivileged_group=None):
    if privileged:
        train_fd = train_fd.get_privileged_group()
        
    elif privileged == False:
        train_fd = train_fd.get_unprivileged_group()

    if not params:
        params = get_model_params(model_type)

    model = train_model(model_type, train_fd, params)
    results = get_classifier_metrics(model, test_fd,
                                     verbose=False,
                                     sel_rate=pos_rate)

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
        # params = {'priors':[0.1, 0.9]}
        params = {}
    else:
        params = {}
    return params


def di_theta(delta_mu_c, delta_mu_a, sigma_u, sigma_p):
    num = 2 
    num += erf((delta_mu_c - delta_mu_a)/(sqrt(2)*sigma_u))
    num -= erf((delta_mu_c + delta_mu_a)/(sqrt(2)*sigma_u))
    
    denom = 2
    denom += erf((delta_mu_c + delta_mu_a)/(sqrt(2)*sigma_p))
    denom -= erf((delta_mu_c - delta_mu_a)/(sqrt(2)*sigma_p))
    
    return num/denom


def di_theta_u(delta_mu_c, delta_mu_a, sigma_u, sigma_p):
    num = 2 
    
    denom = 2
    denom += erf((delta_mu_c + 2 * delta_mu_a)/(sqrt(2)*sigma_p))
    denom -= erf((delta_mu_c - 2 * delta_mu_a)/(sqrt(2)*sigma_p))
    
    return num/denom
    

def report(delta_mu_c, delta_mu_a, sigma_u, sigma_p, verbose=True):
    if verbose:
        print((2+erf((delta_mu_c - delta_mu_a)/(sqrt(2)*sigma_u)) - 
               erf((delta_mu_c + delta_mu_a)/(sqrt(2)*sigma_u)))/4)
        print(1)
        print('Positive Prediction Rate in Privileged Group (Optimal Classifier)')
        print((2+erf((delta_mu_c + delta_mu_a)/(sqrt(2)*sigma_p)) - 
               erf((delta_mu_c - delta_mu_a)/(sqrt(2)*sigma_p)))/4)
        print('Positive Prediction Rate in Unprivileged Group (Unprivileged Classifier)')
        print((2 + erf((delta_mu_c)/(sqrt(2)*sigma_p)) - 
               erf((delta_mu_c)/(sqrt(2)*sigma_p)))/4)
        print('Positive Prediction Rate in Privileged Group (Unprivileged Classifier)')
        print((2 + erf((delta_mu_c + 2 * delta_mu_a)/(sqrt(2)*sigma_p)) - 
               erf((delta_mu_c - 2 * delta_mu_a)/(sqrt(2)*sigma_p)))/4)
    print('DI(theta_u)')
    print(di_theta_u(delta_mu_c, delta_mu_a, sigma_u, sigma_p))
    print('DI(theta)')
    print(di_theta(delta_mu_c, delta_mu_a, sigma_u, sigma_p))

    
def plot_normal(mu, sigma, label=None):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label=label)
    
def plot_non_linear_boundary(mu1, mu2, sigma1, sigma2, p, d, label=None):
    x = np.linspace(-200, 200, 10000)
    y = np.log(p/(1-p)) - d*np.log(sigma1/sigma1) 
    y -= 1/(2*sigma1**2)*(x-mu1)**2 
    y += 1/(2*sigma2**2)*(x-mu2)**2
    plt.plot(x, y, label=label)
    plt.legend()


def get_c123(sigma_1, sigma_2, delta):
    sigma_1_theta_sqr = sigma_1 ** 2 + delta ** 2 / 16
    sigma_2_theta_sqr = sigma_2 ** 2 + delta ** 2 / 16

    denominator = sigma_1_theta_sqr ** 2 * sigma_2 ** 2
    denominator += sigma_2_theta_sqr ** 2 * sigma_1 ** 2
    denominator = 2 * np.sqrt(2) * delta * np.sqrt(denominator)
    c1 = delta ** 2 * sigma_2_theta_sqr / denominator
    c3 = delta ** 2 * sigma_1_theta_sqr / denominator
    c2 = 4 * sigma_1_theta_sqr * sigma_2_theta_sqr / denominator

    return c1, c2, c3


def get_selection_rate(sigma_1, delta, r, alpha, priv, is_tp):
    sigma_2 = r * sigma_1

    c_alpha = np.log(alpha / (1 - alpha))
    c1, c2, c3 = get_c123(sigma_1, sigma_2, delta)
    c = c1 if priv else c3
    c = c + c2 * c_alpha if is_tp else c - c2 * c_alpha

    return 0.5 + (1 if is_tp else -1) * 0.5 * erf(c)


def get_predictions(model, test_fd):
    test_fd_x, test_fd_y = test_fd.get_xy(keep_protected=False)
    return model.predict(test_fd_x)


def  get_model_performances(model, test_fd, pred_func, **kwargs):
    data = test_fd.copy()
    data_pred = test_fd.copy()
    data_pred.labels = pred_func(model, test_fd, **kwargs)

    metrics = ClassificationMetric(
        data, data_pred, privileged_groups=test_fd.privileged_groups,
        unprivileged_groups=test_fd.unprivileged_groups)

    perf = {}
    perf['SR'] = metrics.selection_rate()

    perf['AC_p'] = metrics.accuracy(privileged=True)
    perf['SR_p'] = metrics.selection_rate(privileged=True)
    perf['TPR_p'] = metrics.true_positive_rate(privileged=True)
    perf['FPR_p'] = metrics.false_positive_rate(privileged=True)

    perf['AC_u'] = metrics.accuracy(privileged=False)
    perf['SR_u'] = metrics.selection_rate(privileged=False)
    perf['TPR_u'] = metrics.true_positive_rate(privileged=False)
    perf['FPR_u'] = metrics.false_positive_rate(privileged=False)
    for k in perf:
        perf[k] = perf[k] * 100
    return perf


def print_table_row(is_header=False, alpha=None, p_perf=None,
                    u_perf=None, m_perf=None):
    cols = ["\u03B1", "AC_p", "AC_u", "SR_p", "SR_u", "FPR_p", "FPR_u"]
    if is_header:
        print("\t".join(cols))
    else:
        row = ['{:.2f}'.format(alpha)]
        row += ["{:04.1f}".format(d) for d in [p_perf['AC_p'], u_perf['AC_u']]]
        row += ["{:04.1f}".format(m_perf[c]) for c in cols[3:]]
        print("\t".join(row))
