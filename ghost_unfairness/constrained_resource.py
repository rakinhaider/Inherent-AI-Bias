import sys

sys.path.insert(1, "../")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB

from ghost_unfairness.utils import *


def print_model_performances(model, test_fd, res_constraint=None):
    test_fd_x, test_fd_y = test_fd.get_xy(keep_protected=False)
    data = test_fd.copy()
    data_pred = test_fd.copy()

    if res_constraint:
        favourable_index = np.where(model.classes_ == test_fd.favorable_label)
        predict_proba = model.predict_proba(test_fd_x)[:,favourable_index[0][0]]
        indices = np.argsort(predict_proba)
        indices = indices[-int(np.ceil(len(predict_proba) * res_constraint)):]
        predictions = np.zeros(len(predict_proba))
        predictions[indices] = 1
        data_pred.labels = mod_pred = predictions
    else:
        data_pred.labels = mod_pred = model.predict(test_fd_x)

    # print(mod_pred)
    metrics = ClassificationMetric(data, data_pred,
                privileged_groups=test_fd.privileged_groups,
                unprivileged_groups=test_fd.unprivileged_groups)

    print(metrics.binary_confusion_matrix())
    print('SR\t', metrics.selection_rate())

    print('PCNFM\t', metrics.binary_confusion_matrix(privileged=True))
    print('PSR\t', metrics.selection_rate(privileged=True))
    print('PFPR\t', metrics.false_positive_rate(privileged=True))
    print('PFDR\t', metrics.false_discovery_rate(privileged=True))
    print('UCNFM\t', metrics.binary_confusion_matrix(privileged=False))
    print('USR\t', metrics.selection_rate(privileged=False))
    print('UFPR\t', metrics.false_positive_rate(privileged=False))
    print('UFDR\t', metrics.false_discovery_rate(privileged=False))
    return metrics, mod_pred


if __name__ == "__main__":
    protected = ["sex"]
    privileged_classes = [['Male']]
    metadata = default_mappings.copy()
    metadata['protected_attribute_maps'] = [{1.0: 'Male', 0.0: 'Female'}]

    alpha = 0.5
    beta = 1

    kwargs = {'protected_attribute_names': protected,
              'privileged_classes': [['Male']],
              'metadata': metadata,
              'favorable_classes': [1],
              'beta': beta,
              'alpha': alpha,
              'ds': True
              }

    random_state = 47

    dist = {
        'mu_ps': {'p': 13, 'u': 10},
        'sigma_ps': {'p': 2, 'u': 5},
        'mu_ns': {'p': 3, 'u': 0},
        'sigma_ns': {'p': 2, 'u': 5}
    }

    temp_dist = deepcopy(dist)
    kwargs['dist'] = temp_dist
    print(kwargs['dist'])
    model_type = GaussianNB

    res_constraints = {'Low': 0.1, 'High': 0.9}
    constraint = 'High'

    for alpha in [0.25, 0.5, 0.75]:
        print('alpha', alpha)
        kwargs['alpha'] = alpha
        kwargs['verbose'] = False
        train_fd, test_fd = get_datasets(10000, 2, 1, kwargs,
                                         test_random_state=23)
        pmod, pmod_results = get_groupwise_performance(train_fd, test_fd,
                                            model_type, privileged=True)
        umod, umod_results = get_groupwise_performance(train_fd, test_fd,
                                            model_type, privileged=False)
        mod, _ = get_groupwise_performance(train_fd, test_fd,
                                           model_type, privileged=None)

        print(test_fd.privileged_groups)

        if True:
            print('Theta')
            print(pmod.theta_)
            print(umod.theta_)
            print(mod.theta_)

            print('Sigma')
            print(pmod.sigma_)
            print(umod.sigma_)
            print(mod.sigma_)

            print(mod.class_prior_)

        print('pmod_results', *['{:.4f}'.format(i) for i in pmod_results],
              sep='\t')
        print('umod_results', *['{:.4f}'.format(i) for i in umod_results],
              sep='\t')

        print('mod')
        mod_metrics, mod_pred = print_model_performances(
            mod, test_fd, res_constraint=res_constraints[constraint])
        print('Accuracies')
        print(mod_metrics.accuracy(), mod_metrics.accuracy(privileged=True),
              mod_metrics.accuracy(privileged=False))
