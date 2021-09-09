from ghost_unfairness.sampling.synthetic_generator import(
    synthetic, synthetic_favor_unpriv, synthetic_unfavor_priv,
    group_indices
)
from ghost_unfairness.utils import get_groupwise_performance
from aif360.datasets import(
    CompasDataset, GermanDataset, BankDataset, AdultDataset
)
from aif360.metrics import (BinaryLabelDatasetMetric as BM, ClassificationMetric)
import numpy as np
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from aif360.metrics.utils import(
    compute_boolean_conditioning_vector as condition_vec,
    compute_num_pos_neg
)
from ghost_unfairness.fair_dataset import FairDataset
import argparse
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from scipy.stats import norm

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
    # print(indices[:10], len(indices), group, label)
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
    # print('this', dataset_metric.base_rate(privileged=True))
    # print('this', dataset_metric.base_rate(privileged=False))
    # print('this', dataset_metric.base_rate(privileged=None))
    # get_base_rates(fair_real_dataset)
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
    dataset.scores = np.ones_like(dataset_balanced.labels)
    dataset.scores -= dataset_balanced.labels
    li = len(dataset_balanced.instance_names)
    lf = len(dataset_balanced.features)
    ls = len(dataset_balanced.scores)
    assert li == lf and lf == ls


def get_sample_modes(dataset):
    # Assumed only one favorable class.
    favorable_class = dataset.favorable_label
    unfavorable_class = 1 - favorable_class
    df, _ = dataset.convert_to_dataframe()
    grouped = df.groupby(dataset.protected_attribute_names)
    percents = pd.DataFrame()
    for r, grp in grouped:
        percents[r] = grp[dataset.label_names[0]].value_counts()/len(grp)

    # print(percents)
    if percents[1][1] > percents[0][1]:
        # If privileged has higher base_rate
        #   Inflate favored in unprivileged
        #   Inflate unfavored in privileged.
        sample_mode = 2
    else:
        sample_mode = 1
    return sample_mode


def get_dataset(dataset_name):
    if dataset_name == 'compas':
        dataset = CompasDataset(
            protected_attribute_names=['race'],  # this dataset also contains protected
            # attribute for "sex" which we do not
            # consider in this evaluation
            privileged_classes=[['Caucasian']],  # race Caucasian is considered privileged
            # Just to test (2 features, sensitive attribute and target).
            features_to_keep=['priors_count', 'juv_fel_count',
                              'race', 'two_year_recid'],
            # features_to_drop=['personal_status', 'sex', 'c_charge_desc', 'age'],  # ignore sex-related attributes
            # Simple testing
            features_to_drop=['personal_status', 'sex', 'c_charge_desc', 'age',
                              'age_cat', 'c_charge_degree'],
            favorable_classes=[1]
        )
        dataset.labels = 1 - dataset.labels
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


def print_model_performances(model, test_fd, verbose):
    ### MUST REMOVE TO OTHER MODULE ####
    test_fd_x, test_fd_y = test_fd.get_xy(keep_protected=False)
    data = test_fd.copy()
    data_pred = test_fd.copy()
    data_pred.labels = mod_pred = model.predict(test_fd_x)
    proba = model.predict_proba(test_fd_x)
    # print(model.sigma_)
    # print(model.theta_)
    # print(proba)
    metrics = ClassificationMetric(data,
                                   data_pred,
                                   privileged_groups=test_fd.privileged_groups,
                                   unprivileged_groups=test_fd.unprivileged_groups)

    # print(data.labels.shape)
    # print(data.labels)
    # print(data.labels[0:5, 0])
    # print(data_pred.labels[0:5])
    merged = np.vstack([data.labels[:, 0], data_pred.labels]).transpose()
    # print(merged.shape)
    # print(merged[0:5, :])
    if verbose:
        print(np.unique(merged, axis=0, return_counts=True))

        print(metrics.binary_confusion_matrix())
        print('SR\t', metrics.selection_rate())

        print('PCNFM\t', metrics.binary_confusion_matrix(privileged=True))
        print('PSR\t', metrics.selection_rate(privileged=True))
        print('PTPR\t', metrics.true_positive_rate(privileged=True))
        print('PFPR\t', metrics.false_positive_rate(privileged=True))
        # print('PFDR\t', metrics.false_discovery_rate(privileged=True))
        print('UCNFM\t', metrics.binary_confusion_matrix(privileged=False))
        print('USR\t', metrics.selection_rate(privileged=False))
        print('UTPR\t', metrics.true_positive_rate(privileged=False))
        print('UFPR\t', metrics.false_positive_rate(privileged=False))
        # print('UFDR\t', metrics.false_discovery_rate(privileged=False))
    return metrics, mod_pred


def get_base_rates(dataset):
    df, _ = dataset.convert_to_dataframe()
    grouped = df.groupby(dataset.protected_attribute_names)
    for r, grp in grouped:
        print(r, grp[dataset.label_names].value_counts()/len(grp))
        print(grp[dataset.label_names].value_counts())
        print(len(grp))


def describe(dataset):
    df, _ = dataset.convert_to_dataframe()
    summary = df.describe()
    if debug:
        columns = ['juv_fel_count', 'priors_count']
        print(summary.loc[['mean', 'std']][columns])
    else:
        columns = dataset.feature_names
    grouped = df.groupby(['race'])
    for r, grp in grouped:
        stats = {}
        pos = grp[dataset.label_names[0]] == dataset.favorable_label
        pos_summary = grp[pos].describe()
        neg = grp[dataset.label_names[0]] == dataset.unfavorable_label
        neg_summary = grp[neg].describe()

        for i, c in enumerate(columns):
            key = 'mu_{}_{}_plus'.format(i + 1, 'p' if r else 'u')
            stats[key] = pos_summary.loc['mean'][c]
            key = 'sigma_{}_{}_plus'.format(i + 1, 'p' if r else 'u')
            stats[key] = pos_summary.loc['std'][c]
            key = 'mu_{}_{}_minus'.format(i + 1, 'p' if r else 'u')
            stats[key] = neg_summary.loc['mean'][c]
            key = 'sigma_{}_{}_minus'.format(i + 1, 'p' if r else 'u')
            stats[key] = neg_summary.loc['std'][c]

        mu_stats = [c for c in stats.keys() if c.startswith('mu')]
        sig_stats = [c for c in stats.keys() if c.startswith('sigma')]
        print(*mu_stats, sep='\t')
        print(*['{:10.4f}'.format(stats[c]) for c in mu_stats], sep='\t')
        print(*sig_stats, sep='\t')
        print(*['{:10.4f}'.format(stats[c]) for c in sig_stats], sep='\t')


def scoring_func(y, y_pred, privileged, data):
    data = data.subset(range(len(y_pred)))
    pred = data.copy(deepcopy=True)
    pred.labels = y_pred
    data.labels = y

    metric = ClassificationMetric(data, pred,
                                  privileged_groups=data.privileged_groups,
                                  unprivileged_groups=data.unprivileged_groups)
    return metric.selection_rate(privileged)


if __name__ == "__main__":
    np.random.seed(23)

    args = argparse.ArgumentParser(
            description="Homework 2",
    )
    args.add_argument("-d", "--data",
              choices=["compas", "german", "bank"],
              default='compas'
    )
    args.add_argument('-k', '--k-fold',
                      default=20
                      )
    args.add_argument('-m', '--model-type',
                      default='nb'
                      )
    args = args.parse_args()

    debug = True

    dataset_orig = get_dataset(args.data)

    sample_mode = 2

    if args.model_type == 'nb':
        model_type = GaussianNB
    elif args.model_type == 'svm':
        model_type = SVC
    elif args.model_type == 'lr':
        model_type = LogisticRegression
    else:
        raise NotImplementedError('Classifier not supported.')

    # get_base_rates(dataset_orig)
    p_group = [{'race': 1}]
    u_group = [{'race': 0}]
    # NB Performance on dataset_orig
    train_orig, test_orig = dataset_orig.split([0.7], shuffle=True)
    train_orig_x, train_orig_y = train_orig.features, train_orig.labels
    train_orig_x = train_orig_x[:, 1:]
    test_orig_x, test_orig_y = test_orig.features, test_orig.labels
    test_orig_x = test_orig_x[:, 1:]

    mod = GaussianNB()
    mod.fit(train_orig_x, train_orig_y)
    test_orig_pred = test_orig.copy()
    test_orig_pred.labels = mod.predict(test_orig_x)
    metric = ClassificationMetric(test_orig, test_orig_pred,
                                  privileged_groups=p_group,
                                  unprivileged_groups=u_group)

    # get_base_rates(dataset_balanced)

    dataset_balanced = get_balanced_dataset(dataset_orig, sample_mode=2)
    fix_balanced_dataset(dataset_balanced)
    # get_base_rates(dataset_balanced)
    columns = ['mean_difference', 'disparate_impact',
               'Opt_Acc', 'Priv_Acc', 'Unpriv_Acc',
               'Priv_True_pos', 'Priv_False_pos',
               'Unpriv_True_pos', 'Unpriv_False_pos',
               ]
    results = pd.DataFrame(columns=columns)

    args.k_fold = 1
    for alpha in [0.25, 0.5, 0.75, 0.9]:
        # print(alpha)
        fd = get_real_fd(dataset_balanced, alpha=alpha)
        fold_metrics = []
        for i in [23, 29, 37, 41, 47]:
            fd_train, fd_test = fd.split([0.7], shuffle=True)

            # mod, m_result = get_groupwise_performance(
            #     fd_train, fd_test, model_type, privileged=None, pos_rate=True)

            x, y = fd_train.get_xy(keep_protected=False)
            mod = model_type()
            mod.fit(x, y)

            metrics, mod_pred = print_model_performances(mod, fd_test, True)
            fold_metrics.append(metrics)

        sel_rates = [fr.selection_rate(privileged=None) for fr in fold_metrics]
        p_sel_rates = [fr.selection_rate(privileged=True) for fr in fold_metrics]
        u_sel_rates = [fr.selection_rate(privileged=False) for fr in fold_metrics]
        print(sel_rates)
        print(np.mean(sel_rates))
        print(np.std(sel_rates))

        print(p_sel_rates)
        print(np.mean(p_sel_rates))
        print(np.std(p_sel_rates))

        print(u_sel_rates)
        print(np.mean(u_sel_rates))
        print(np.std(u_sel_rates))
        # break

    title = '_'.join([args.data, args.model_type])
    results.to_csv('outputs/' + title + '.tsv', sep='\t')