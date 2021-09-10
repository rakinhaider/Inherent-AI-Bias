import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from ghost_unfairness.utils import *
from sklearn.naive_bayes import GaussianNB
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma-1', default=2, type=int)
    parser.add_argument('--sigma-2', default=5, type=int)
    parser.add_argument('--mu_p_plus', default=13, type=int)
    parser.add_argument('--mu_u_plus', default=10, type=int)
    parser.add_argument('--delta', default=10, type=int)
    parser.add_argument('--beta', default=1, type=int)
    parser.add_argument('--n-samples', default=10000, type=int)
    parser.add_argument('--n-redline', default=1, type=int)
    parser.add_argument('--n-feature', default=0, type=int)
    parser.add_argument('--tr-rs', default=47, type=int,
                        help='Train Random Seed')
    parser.add_argument('--te-rs', default=41, type=int,
                        help='Test Random Seed')
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
    model_type = GaussianNB
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
