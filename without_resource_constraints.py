from inherent_bias.utils import *
from utils import get_parser, get_estimator

if __name__ == "__main__":
    args = get_parser().parse_args()

    protected = ["sex"]
    privileged_classes = [['Male']]
    metadata = default_mappings.copy()
    metadata['protected_attribute_maps'] = [{1.0: 'Male', 0.0: 'Female'}]

    kwargs = {'protected_attribute_names': protected,
              'privileged_classes': [['Male']],
              'metadata': metadata,
              'favorable_classes': [1],
              'beta': args.beta,
              'ds': True,
              'n_dep_feat': 0
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

    print_table_row(is_header=True)
    for alpha in [0.25, 0.5, 0.75]:
        kwargs['alpha'] = alpha
        kwargs['verbose'] = False
        train_fd, test_fd = get_datasets(n_samples, n_feature, n_redline,
                                         kwargs, train_random_state=args.tr_rs,
                                         test_random_state=args.te_rs)
        pmod, pmod_results = get_groupwise_performance(train_fd, test_fd,
                                                       estimator,
                                                       privileged=True,
                                                       pos_rate=False)
        umod, umod_results = get_groupwise_performance(train_fd, test_fd,
                                                       estimator,
                                                       privileged=False,
                                                       pos_rate=False)
        mod, mod_results = get_groupwise_performance(train_fd, test_fd,
                                                     estimator,
                                                     privileged=None,
                                                     pos_rate=False)

        p_perf = get_model_performances(pmod, test_fd,
                                        get_predictions, keep_prot=keep_prot)
        u_perf = get_model_performances(umod, test_fd,
                                        get_predictions, keep_prot=keep_prot)
        m_perf = get_model_performances(mod, test_fd,
                                        get_predictions, keep_prot=keep_prot)
        print_table_row(is_header=False, alpha=alpha, p_perf=p_perf,
                        u_perf=u_perf, m_perf=m_perf)
