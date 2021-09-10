from ghost_unfairness.utils import *
from sklearn.naive_bayes import GaussianNB


def print_model_performances(model, test_fd):
    test_fd_x, test_fd_y = test_fd.get_xy(keep_protected=False)
    data = test_fd.copy()
    data_pred = test_fd.copy()

    data_pred.labels = mod_pred = model.predict(test_fd_x)
    metrics = ClassificationMetric(data,
                                   data_pred,
                                   privileged_groups=test_fd.privileged_groups,
                                   unprivileged_groups=test_fd.unprivileged_groups)



    print('SR\t', metrics.selection_rate())

    print('PSR\t', metrics.selection_rate(privileged=True))
    print('PTPR\t', metrics.true_positive_rate(privileged=True))
    print('PFPR\t', metrics.false_positive_rate(privileged=True))
    print('USR\t', metrics.selection_rate(privileged=False))
    print('UTPR\t', metrics.true_positive_rate(privileged=False))
    print('UFPR\t', metrics.false_positive_rate(privileged=False))
    return metrics, mod_pred


if __name__ == "__main__":

    protected = ["sex"]
    privileged_classes = [['Male']]
    metadata = default_mappings.copy()
    metadata['protected_attribute_maps'] = [{1.0: 'Male', 0.0: 'Female'}]

    beta = 1

    kwargs = {'protected_attribute_names': protected,
              'privileged_classes': [['Male']],
              'metadata': metadata,
              'favorable_classes': [1],
              'beta': beta,
              'ds': True,
              'n_dep_feat': 0
              }

    train_random_state = 47
    test_random_state = 41

    dist = {
        'mu_ps': {'p': 13, 'u': 10},
        'sigma_ps': {'p': 13, 'u': 25},
        'mu_ns': {'p': 3, 'u': 0},
        'sigma_ns': {'p': 13, 'u': 25}
    }

    temp_dist = deepcopy(dist)
    kwargs['dist'] = temp_dist
    model_type = GaussianNB
    n_redline = 1
    n_feature = 0

    print_table_row(is_header=True)

    for alpha in [0.25, 0.5, 0.75]:
        kwargs['alpha'] = alpha
        kwargs['shift_random'] = 0
        kwargs['shift_priv'] = False
        kwargs['verbose'] = False
        train_fd, test_fd = get_datasets(100000, n_feature, n_redline, kwargs,
                                         train_random_state=train_random_state,
                                         test_random_state=test_random_state)
        pmod, pmod_results = get_groupwise_performance(train_fd, test_fd,
                                                       model_type,
                                                       privileged=True,
                                                       pos_rate=False)
        umod, umod_results = get_groupwise_performance(train_fd, test_fd,
                                                       model_type,
                                                       privileged=False,
                                                       pos_rate=False)
        mod, mod_results = get_groupwise_performance(train_fd, test_fd,
                                                     model_type,
                                                     privileged=None,
                                                     pos_rate=False)


        p_perf = get_model_performances(pmod, test_fd, get_predictions)
        u_perf = get_model_performances(umod, test_fd, get_predictions)
        m_perf = get_model_performances(mod, test_fd, get_predictions)
        print_table_row(is_header=False, alpha=alpha, p_perf=p_perf,
                        u_perf=u_perf, m_perf=m_perf)

