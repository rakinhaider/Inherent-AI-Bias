import os
import matplotlib.pyplot as plt
from inherent_bias.utils import *
from utils import get_parser, get_estimator
from .plot_decision_boundary import plot_decision_boundaries

# Suppresing tensorflow warning
warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument('--alpha', default=.25, type=float)
    parser.add_argument('--filetype', default='pdf', type=str)
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
    print(dist)
    temp_dist = deepcopy(dist)
    kwargs['dist'] = temp_dist
    estimator = get_estimator(args.estimator, args.reduce)
    keep_prot = args.reduce or (args.estimator == 'pr')
    n_samples = args.n_samples
    n_redline = args.n_redline
    n_feature = args.n_feature

    kwargs['alpha'] = args.alpha
    kwargs['verbose'] = False
    train_fd, test_fd = get_datasets(n_samples, n_feature, n_redline,
                                     kwargs, train_random_state=args.tr_rs,
                                     test_random_state=args.te_rs)

    df, y = train_fd.get_xy(keep_protected=True)
    df['labels'] = y
    for (s, label), grp in df.groupby(['sex', 'labels']):
        plt.scatter(grp['r_p_0'], grp['r_u_0'])

    mod, mod_results = get_groupwise_performance(train_fd, test_fd,
                                                 estimator,
                                                 privileged=None,
                                                 pos_rate=False)


    plot_decision_boundaries(df, y, df['sex'], GaussianNB,
                             plot_by_group=True, dist=dist, delta=args.delta)

    plt.savefig('./poster/figure/model_boundary_{}.{}'.format(
        args.alpha, args.filetype), format=args.filetype)