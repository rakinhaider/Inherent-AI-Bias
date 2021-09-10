import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from inherent_bias.sampling.synthetic_generator import(
    synthetic
)
from aif360.datasets import(
    GermanDataset, BankDataset
)
from aif360.algorithms.preprocessing.optim_preproc_helpers.\
    data_preproc_functions import load_preproc_data_compas
from aif360.metrics import (BinaryLabelDatasetMetric as BM)
from sklearn.svm import SVC
from aif360.metrics.utils import(
    compute_boolean_conditioning_vector as condition_vec,
    compute_num_pos_neg
)
import argparse
from inherent_bias.mixed_model_nb import MixedModelNB
from inherent_bias.utils import *


def get_group_dicts(dataset):
    if len(dataset.protected_attribute_names):
        p_attr = dataset.protected_attribute_names[0]
        return [{p_attr: 1}], [{p_attr: 0}]
    else:
        raise NotImplementedError("Not implemented.")


def get_indices(dataset, condition=None, label=None):
    cond_vec = condition_vec(dataset.features,
                             dataset.feature_names, condition)

    if label is not None:
        truth_values = dataset.labels == label
        truth_values = truth_values.reshape(-1)
        cond_vec = np.logical_and(cond_vec, truth_values)

    return [i for i, val in enumerate(cond_vec) if val]


def get_pos_neg_by_group(dataset, group):
    pos = compute_num_pos_neg(dataset.features, dataset.labels,
                              dataset.instance_weights,
                              dataset.feature_names,
                              dataset.favorable_label, group)

    neg = compute_num_pos_neg(dataset.features, dataset.labels,
                              dataset.instance_weights,
                              dataset.feature_names,
                              dataset.unfavorable_label, group)
    return pos, neg


def get_samples_by_group(dataset, n_samples, group, label):
    indices = get_indices(dataset, group, label)
    indices = np.random.choice(indices, n_samples, replace=False)
    return indices


def sample_fd_indices(dataset, alpha=0.5, beta=1):
    # First find n_unprivileged
    # Then check number of samples to take from each group
    # Then sample and merge them together.
    p_group, u_group = get_group_dicts(dataset)
    p_pos, p_neg = get_pos_neg_by_group(dataset, p_group)
    u_pos, u_neg = get_pos_neg_by_group(dataset, u_group)

    n_unpriv = min(u_pos/alpha, u_neg/(1 - alpha),
                   p_pos/(alpha * beta), p_neg/((1-alpha)*beta)
    )
    n_priv = beta * n_unpriv

    f_label = dataset.favorable_label
    uf_label = dataset.unfavorable_label

    temp = [(alpha * n_unpriv, u_group, f_label),
            ((1 - alpha) * n_unpriv, u_group, uf_label),
            (alpha * n_priv, p_group, f_label),
            ((1 - alpha) * n_priv, p_group, uf_label)]

    indices = []
    for n, g, f in temp:
        # print(n)
        sample_indices = get_samples_by_group(dataset, round(n), g, f)
        indices.extend(sample_indices)

    return indices


def get_real_fd(dataset, alpha=0.5, beta=1):
    indices = sample_fd_indices(dataset, alpha, beta)
    fair_real_dataset = dataset.subset(indices)
    u_group, p_group = get_group_dicts(dataset)
    dataset_metric = BM(fair_real_dataset, u_group, p_group)
    assert abs(dataset_metric.base_rate(privileged=True) - alpha) < 1e-3
    assert abs(dataset_metric.base_rate(privileged=False) - alpha) < 1e-3
    assert abs(dataset_metric.base_rate(privileged=None) - alpha) < 1e-3
    fd = FairDataset(2, 1, 1)
    fd.update_from_dataset(fair_real_dataset)
    return fd


def get_balanced_dataset(dataset, sample_mode):
    f_label = dataset.favorable_label
    uf_label = dataset.unfavorable_label
    p_group, u_group = get_group_dicts(dataset)

    dataset_metric = BM(dataset, u_group, p_group)

    base_rate_p = dataset_metric.base_rate(privileged=True)
    base_rate_u = dataset_metric.base_rate(privileged=False)

    dataset = synthetic(dataset, u_group,
                        base_rate_p, base_rate_u,
                        f_label, uf_label,
                        None, sample_mode, 1.00)

    return dataset


def fix_balanced_dataset(dataset):
    n = len(dataset.features)
    # Fixing balanced dataset attributes
    dataset.instance_names = [str(i) for i in range(n)]
    dataset.scores = np.ones_like(dataset.labels)
    dataset.scores -= dataset.labels
    li = len(dataset.instance_names)
    lf = len(dataset.features)
    ls = len(dataset.scores)
    assert li == lf and lf == ls


def get_dataset(dataset_name):
    if dataset_name == 'compas':
        dataset = load_preproc_data_compas(protected_attributes=['race'])
    elif dataset_name == 'german':
        dataset = GermanDataset(
            # this dataset also contains protected
            # attribute for "sex" which we do not
            # consider in this evaluation
            protected_attribute_names=['age'],
            # age >=25 is considered privileged
            privileged_classes=[lambda x: x >= 25],
            # ignore sex-related attributes
            features_to_drop=['personal_status', 'sex'],
        )
    elif dataset_name == 'bank':
        dataset = BankDataset(
            # this dataset also contains protected
            protected_attribute_names=['age'],
            privileged_classes=[lambda x: x >= 25],  # age >=25 is considered privileged
        )
        dataset.metadata['protected_attribute_maps'] = [{1.0: 'yes', 0.0: 'no'}]
        temp = dataset.favorable_label
        dataset.favorable_label = dataset.unfavorable_label
        dataset.unfavorable_label = temp
    else:
        raise ValueError('Dataset name must be one of '
                         'compas, german, bank')
    return dataset


def exp_compas_orig(params, verbose=False):
    train_orig, test_orig = dataset_orig.split([0.8], shuffle=True)
    params['bern_indices'] = [i - 1 for i in params['bern_indices']]
    params['geom_indices'] = [i - 1 for i in params['geom_indices']]
    train_orig_fd = FairDataset(100, 0, 1)
    train_orig_fd.update_from_dataset(train_orig)
    test_orig_fd = FairDataset(100, 0, 1)
    test_orig_fd.update_from_dataset(test_orig)
    pmod, p_result = get_groupwise_performance(
        train_orig_fd, test_orig_fd, MixedModelNB,
        privileged=True, params=params, pos_rate=True
    )

    umod, u_result = get_groupwise_performance(
        train_orig_fd, test_orig_fd, MixedModelNB,
        privileged=False, params=params, pos_rate=True
    )
    mod, m_result = get_groupwise_performance(
        train_orig_fd, test_orig_fd, GaussianNB,
        privileged=None, params={}, pos_rate=True
    )

    # Orig data results are not generated from here.
    if verbose:
        print('Orig group-wise')
        print(*['{:.4f}'.format(i) for i in p_result], sep='\t')
        print(*['{:.4f}'.format(i) for i in u_result], sep='\t')
        print(*['{:.4f}'.format(i) for i in m_result], sep='\t')
        # print_model_performances(mod, test_orig_fd, verbose=True)


if __name__ == "__main__":
    np.random.seed(23)

    args = argparse.ArgumentParser(
            description="OversampleCompasExperiment",
    )
    args.add_argument("-d", "--data", choices=["compas"], default='compas')
    args.add_argument('-m', '--model-type', choices=['gnb', 'mixednb'],
                      default='mixednb')
    args.add_argument('-s', '--sample-mode', default=2, type=int)
    args.add_argument('--split', default=0.7, help='Train test split')
    args.add_argument('-r', '--random-seed', default=23)

    args = args.parse_args()
    sample_mode = args.sample_mode
    random_seed = args.random_seed
    dataset_orig = get_dataset(args.data)

    if args.model_type == 'gnb':
        model_type = GaussianNB
    elif args.model_type == 'mixednb':
        model_type = MixedModelNB
    elif args.model_type == 'svm':
        model_type = SVC
    elif args.model_type == 'lr':
        model_type = LogisticRegression
    else:
        raise NotImplementedError('Classifier not supported.')

    c_charge_desc_cols = [c for c in dataset_orig.feature_names
                          if c.startswith('c_charge_desc')]
    bern_col = ['age_cat=25 to 45', 'age_cat=Greater than 45',
                'age_cat=Less than 25',
                'c_charge_degree=F', 'c_charge_degree=M',
                'priors_count=0', 'priors_count=1 to 3',
                'priors_count=More than 3'] + c_charge_desc_cols
    bern_indices = [list(dataset_orig.feature_names).index(c)
                    for c in bern_col]

    geom_col = []
    geom_indices = [list(dataset_orig.feature_names).index(c)
                    for c in geom_col]
    params = {'alpha': 1, 'bern_indices': bern_indices,
              'geom_indices': geom_indices}
    # The following line is not necessary here. Didn't remove to
    # ensure reproducibility. Inside the function we used random numbers.
    # Removing them will  result different random numbers in later experiments.
    # Similar pattern in results but different values.
    exp_compas_orig(params, verbose=False)

    dataset_balanced = get_balanced_dataset(dataset_orig,
                                            sample_mode=sample_mode)
    fix_balanced_dataset(dataset_balanced)

    params = {'alpha': 1}
    print_table_row(is_header=True)
    for alpha in [0.25, 0.5, 0.75]:
        fd = get_real_fd(dataset_balanced, alpha=alpha)
        train_fd, test_fd = fd.split([args.split], shuffle=True,
                                     seed=args.random_seed)

        train_x, train_y = train_fd.get_xy(keep_protected=False)
        bern_indices = [list(train_x.columns).index(c)
                        for c in bern_col]
        geom_indices = [list(train_x.columns).index(c) for c in
                        geom_col]
        params['bern_indices'] = bern_indices
        params['geom_indices'] = geom_indices

        pmod, p_result = get_groupwise_performance(
            train_fd, test_fd, model_type,
            privileged=True, params=params, pos_rate=False
        )

        umod, u_result = get_groupwise_performance(
            train_fd, test_fd, model_type,
            privileged=False, params=params, pos_rate=False
        )
        mod, m_result = get_groupwise_performance(
            train_fd, test_fd, model_type,
            privileged=None, params=params, pos_rate=False
        )
        p_perf = get_model_performances(pmod, test_fd, get_predictions)
        u_perf = get_model_performances(umod, test_fd, get_predictions)
        m_perf = get_model_performances(mod, test_fd, get_predictions)
        print_table_row(is_header=False, alpha=alpha, p_perf=p_perf,
                        u_perf=u_perf, m_perf=m_perf)
