from inherent_bias.utils import *
from utils import get_parser, get_estimator


def print_model_performances(model, test_fd, res_constraint=None,
                             print_acc=False):
    test_fd_x, test_fd_y = test_fd.get_xy(keep_protected=False)
    data = test_fd.copy()
    data_pred = test_fd.copy()

    if res_constraint:
        predictions = get_constrained_predictions(
            model, test_fd, res_constraint=res_constraint)
        data_pred.labels = mod_pred = predictions
    else:
        data_pred.labels = mod_pred = model.predict(test_fd_x)

    metrics = ClassificationMetric(data, data_pred,
                privileged_groups=test_fd.privileged_groups,
                unprivileged_groups=test_fd.unprivileged_groups)

    print(metrics.binary_confusion_matrix())
    print('SR\t', metrics.selection_rate())

    if print_acc:
        print('PACC\t', metrics.accuracy(privileged=True))
    print('PSR\t', metrics.selection_rate(privileged=True))
    print('PFPR\t', metrics.false_positive_rate(privileged=True))
    print('PFDR\t', metrics.false_discovery_rate(privileged=True))
    if print_acc:
        print('UACC\t', metrics.accuracy(privileged=False))
    print('USR\t', metrics.selection_rate(privileged=False))
    print('UFPR\t', metrics.false_positive_rate(privileged=False))
    print('UFDR\t', metrics.false_discovery_rate(privileged=False))
    return metrics, mod_pred


def get_constrained_predictions(model, test_fd, keep_prot, res_constraint):
    test_fd_x, test_fd_y = test_fd.get_xy(keep_protected=keep_prot)
    favourable_index = np.where(model.classes_ == test_fd.favorable_label)
    predict_proba = model.predict_proba(test_fd_x)[:, favourable_index[0][0]]
    indices = np.argsort(predict_proba)
    indices = indices[-int(np.ceil(len(predict_proba) * res_constraint)):]
    predictions = np.zeros(len(predict_proba))
    predictions[indices] = 1
    return predictions


if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument('-r', '--resource', choices=['Low', 'High'],
                        default='Low')
    args = parser.parse_args()

    protected = ["sex"]
    privileged_classes = [['Male']]
    metadata = default_mappings.copy()
    metadata['protected_attribute_maps'] = [{1.0: 'Male', 0.0: 'Female'}]

    kwargs = {'protected_attribute_names': protected,
              'privileged_classes': [['Male']],
              'metadata': metadata,
              'favorable_classes': [1],
              'beta': args.beta,
              'ds': True
              }

    dist = {
        'mu_ps': {'p': args.mu_p_plus, 'u': args.mu_u_plus},
        'sigma_ps': {'p': args.sigma_1, 'u': args.sigma_2},
        'mu_ns': {'p': args.mu_p_plus - args.delta,
                  'u': args.mu_u_plus - args.delta},
        'sigma_ns': {'p': args.sigma_1, 'u': args.sigma_2}
    }

    temp_dist = deepcopy(dist)
    kwargs['dist'] = temp_dist
    estimator = get_estimator(args.estimator, args.reduce)
    keep_prot = args.reduce or (args.estimator == 'pr')
    n_samples = args.n_samples
    n_redline = args.n_redline
    n_feature = args.n_feature

    res_constraints = {'Low': 0.1, 'High': 0.9}
    constraint = res_constraints[args.resource]

    print_table_row(is_header=True)
    for alpha in [0.25, 0.5, 0.75]:
        kwargs['alpha'] = alpha
        kwargs['verbose'] = False
        train_fd, test_fd = get_datasets(n_samples, n_feature, n_redline,
                                         kwargs, train_random_state=args.tr_rs,
                                         test_random_state=args.te_rs)
        pmod, pmod_results = get_groupwise_performance(train_fd, test_fd,
                                                       estimator,
                                                       privileged=True)
        umod, umod_results = get_groupwise_performance(train_fd, test_fd,
                                                       estimator,
                                                       privileged=False)
        mod, _ = get_groupwise_performance(train_fd, test_fd,
                                           estimator, privileged=None)

        p_perf = get_model_performances(
            pmod, test_fd, get_constrained_predictions,
            keep_prot=keep_prot, res_constraint=constraint)
        u_perf = get_model_performances(
            umod, test_fd, get_constrained_predictions,
            keep_prot=keep_prot, res_constraint=constraint)
        m_perf = get_model_performances(
            mod, test_fd, get_constrained_predictions,
            keep_prot=keep_prot, res_constraint=constraint)
        print_table_row(is_header=False, alpha=alpha, p_perf=p_perf,
                        u_perf=u_perf, m_perf=m_perf)
