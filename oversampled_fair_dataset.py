from ghost_unfairness.sampling.synthetic_generator import(
    synthetic, synthetic_favor_unpriv, synthetic_unfavor_priv,
    group_indices
)
from aif360.datasets import(
    CompasDataset, GermanDataset, BankDataset, AdultDataset
)
from aif360.metrics import BinaryLabelDatasetMetric as BM
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from aif360.metrics.utils import(
    compute_boolean_conditioning_vector as condition_vec,
    compute_num_pos_neg
)
from ghost_unfairness.utils import get_groupwise_performance
from ghost_unfairness.fair_dataset import FairDataset
import argparse
import pandas as pd


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
    assert dataset_metric.base_rate(privileged=True) == alpha
    assert dataset_metric.base_rate(privileged=False) == alpha
    assert dataset_metric.base_rate(privileged=None) == alpha
    fd = FairDataset(2, 1, 1)
    fd.update_from_dataset(fair_real_dataset)
    return fd


def get_balanced_dataset(dataset, sample_mode=2):
    f_label = dataset.favorable_label
    uf_label = dataset.unfavorable_label
    p_group, u_group = get_group_dicts(dataset)

    dataset_metric = BM(dataset, u_group, p_group)

    base_rate_p = dataset_metric.base_rate(privileged=True)
    base_rate_u = dataset_metric.base_rate(privileged=False)

    dataset_balanced = synthetic(dataset, u_group,
                                 base_rate_p, base_rate_u,
                                 f_label, uf_label,
                                 None, sample_mode, 1.00)

    return dataset_balanced


def fix_balanced_dataset(dataset):
    n = len(dataset.features)
    # Fixing balanced dataset attributes
    dataset.instance_names = [str(i) for i in range(n)]
    dataset.scores = np.ones_like(dataset_balanced.labels)
    dataset.scores -= dataset_balanced.labels
    li = len(dataset_balanced.instance_names)
    lf = len(dataset_balanced.features)
    ls = len(dataset_balanced.scores)
    assert li == lf and lf == ls


def get_dataset(dataset_name):
    if dataset_name == 'compas':
        dataset = CompasDataset(
            protected_attribute_names=['race'],  # this dataset also contains protected
            # attribute for "sex" which we do not
            # consider in this evaluation
            privileged_classes=[['Caucasian']],  # race Caucasian is considered privileged
            features_to_drop=['personal_status', 'sex']  # ignore sex-related attributes
        )

        # dataset.metadata['protected_attribute_maps'] = \
        #     [{1.0: 'Caucasian', 0.0: 'Not Caucasian'}]
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


if __name__ == "__main__":
    np.random.seed(23)

    args = argparse.ArgumentParser(
            description="Homework 2",
    )
    args.add_argument("-d", "--data",
              help="dataset: compas, german, bank",
              default='compas'
    )
    args.add_argument('-k', '--k-fold',
                      default=20
                      )
    args.add_argument('-m', '--model-type',
                      default='nb'
                      )
    args = args.parse_args()

    dataset_orig = get_dataset(args.data)

    sample_mode = 2
    if args.data == 'bank':
        sample_mode = 2

    if args.model_type == 'nb':
        model_type = GaussianNB
    elif args.model_type == 'svm':
        model_type = SVC
    elif args.model_type == 'lr':
        model_type = LogisticRegression

    dataset_balanced = get_balanced_dataset(dataset_orig,
                                            sample_mode=sample_mode)
    fix_balanced_dataset(dataset_balanced)
    fd_train = get_real_fd(dataset_balanced)
    columns = ['mean_difference', 'disparate_impact',
               'Opt_Acc', 'Priv_Acc', 'Unpriv_Acc',
               'Priv_True_pos', 'Priv_False_pos',
               'Unpriv_True_pos', 'Unpriv_False_pos',
               ]
    results = pd.DataFrame(columns=columns)
    for i in range(0, int(args.k_fold)):
        fd_test = get_real_fd(dataset_balanced)
        pmod, p_result = get_groupwise_performance(fd_train, fd_test, model_type,
                                                   privileged=True, pos_rate=True)
        print(*p_result, sep='\t')
        results = results.append(dict(zip(columns, p_result)), ignore_index=True)
        umod, u_result = get_groupwise_performance(fd_train, fd_test, model_type,
                                                   privileged=False, pos_rate=True)
        print(*u_result, sep='\t')
        results = results.append(dict(zip(columns, u_result)), ignore_index=True)

        mod, m_result = get_groupwise_performance(fd_train, fd_test, model_type,
                                                  privileged=None, pos_rate=True)
        print(*m_result, sep='\t')
        results = results.append(dict(zip(columns, m_result)), ignore_index=True)

        if abs(1 - m_result[1]) > abs(1 - u_result[1]) or \
                abs(1 - m_result[1]) > abs(1 - p_result[1]):
            print()
        else:
            print("------------VIOLATION-----------------")

    title = '_'.join([args.data, args.model_type])
    results.to_csv('outputs/' + title + '.csv', sep='\t')
