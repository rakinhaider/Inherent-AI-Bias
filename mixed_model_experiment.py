from ghost_unfairness.sampling.synthetic_generator import(
    synthetic
)
from aif360.datasets import(
    CompasDataset, GermanDataset, BankDataset, AdultDataset
)
from aif360.algorithms.preprocessing.optim_preproc_helpers.\
    data_preproc_functions import load_preproc_data_compas
from aif360.datasets.compas_dataset import default_preprocessing
from aif360.metrics import (BinaryLabelDatasetMetric as BM, ClassificationMetric)
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from aif360.metrics.utils import(
    compute_boolean_conditioning_vector as condition_vec,
    compute_num_pos_neg
)
from ghost_unfairness.fair_dataset import FairDataset
import argparse
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from ghost_unfairness.mixed_model_nb import MixedModelNB
from ghost_unfairness.utils import get_groupwise_performance


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
    dataset.scores = np.ones_like(dataset.labels)
    dataset.scores -= dataset.labels
    li = len(dataset.instance_names)
    lf = len(dataset.features)
    ls = len(dataset.scores)
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


def compas_preprocessing(df):
    label = 'two_year_recid'
    # Preprocess and drops samples not used in propublica analysis.
    # aif360 always applies this default processing.
    df = default_preprocessing(df)
    grouped = df.groupby(by=['race'])
    keep_races = ['African-American', 'Caucasian']
    for r, grp in grouped:
        if r not in keep_races:
            indices = df[df['race'] == r].index
            df = df.drop(index=indices)
    return df


def get_dataset(dataset_name):
    if dataset_name == 'compas':
        dataset = CompasDataset(
            protected_attribute_names=['race'],  # this dataset also contains protected
            # attribute for "sex" which we do not
            # consider in this evaluation
            privileged_classes=[['Caucasian']],  # race Caucasian is considered privileged
            # Just to test (2 features, sensitive attribute and target).
            # features_to_keep=['priors_count', 'juv_fel_count',
            #                  'race', 'two_year_recid'],
            features_to_drop=['personal_status', 'sex'],  # ignore sex-related attributes
            # Simple testing
            # features_to_drop=['personal_status', 'sex', 'c_charge_desc', 'age',
            #                  'age_cat', 'c_charge_degree'],
            favorable_classes=[1],
            custom_preprocessing=compas_preprocessing
        )

        dataset = load_preproc_data_compas(protected_attributes=['race'])
        # dataset.labels = 1 - dataset.labels
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
    metrics = ClassificationMetric(data,
                                   data_pred,
                                   privileged_groups=test_fd.privileged_groups,
                                   unprivileged_groups=test_fd.unprivileged_groups)

    if verbose:
        print(metrics.binary_confusion_matrix())
        print('SR\t', metrics.selection_rate())

        print('PCNFM\t', metrics.binary_confusion_matrix(privileged=True))
        print('PACC\t', metrics.accuracy(privileged=True))
        print('PSR\t', metrics.selection_rate(privileged=True))
        print('PTPR\t', metrics.true_positive_rate(privileged=True))
        print('PFPR\t', metrics.false_positive_rate(privileged=True))
        print('UCNFM\t', metrics.binary_confusion_matrix(privileged=False))
        print('UACC\t', metrics.accuracy(privileged=False))
        print('USR\t', metrics.selection_rate(privileged=False))
        print('UTPR\t', metrics.true_positive_rate(privileged=False))
        print('UFPR\t', metrics.false_positive_rate(privileged=False))
    return metrics, mod_pred


def get_base_rates(dataset):
    df, _ = dataset.convert_to_dataframe()
    print(df['two_year_recid'].value_counts()/len(df))
    grouped = df.groupby(dataset.protected_attribute_names)
    for r, grp in grouped:
        print(r, grp[dataset.label_names].value_counts()/len(grp))
        print(len(grp))


def scoring_func(y, y_pred, privileged, data):
    data = data.subset(range(len(y_pred)))
    pred = data.copy(deepcopy=True)
    pred.labels = y_pred
    data.labels = y

    metric = ClassificationMetric(data, pred,
                                  privileged_groups=data.privileged_groups,
                                  unprivileged_groups=data.unprivileged_groups)
    return metric.selection_rate(privileged)


def random_flips(dataset, privileged=False, n=10):
    group = 1 if privileged else 0
    df, _ = dataset.convert_to_dataframe()
    df['labels'] = dataset.labels
    pos_group = df[(df['race'] == group) & (df['labels'] == 1)]
    neg_group = df[(df['race'] == group) & (df['labels'] == 0)]

    pos_sample = pos_group.sample(n=n)
    neg_sample = neg_group.sample(n=n)

    for i in pos_sample.index:
        df.loc[i]['labels'] = 0
    for i in neg_sample.index:
        df.loc[i]['labels'] = 1
    dataset.labels = df['labels'].values.reshape(-1, 1)
    return dataset


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
    # print(dataset_orig.features.shape)
    sample_mode = get_sample_modes(dataset_orig)

    if args.model_type == 'nb':
        # model_type = GaussianNB
        # model_type = CategoricalNB
        pass
    elif args.model_type == 'svm':
        model_type = SVC
    elif args.model_type == 'lr':
        model_type = LogisticRegression
    else:
        raise NotImplementedError('Classifier not supported.')

    p_group = [{'race': 1}]
    u_group = [{'race': 0}]
    # get_base_rates(dataset_orig)
    train_orig, test_orig = dataset_orig.split([0.8], shuffle=True)
    train_orig_x, _ = train_orig.convert_to_dataframe()
    columns = list(train_orig_x.columns)
    columns = columns[0:1] + columns[2:-1]
    train_orig_y = train_orig.labels.ravel()
    train_orig_x = train_orig_x[columns]
    test_orig_x, _ = test_orig.convert_to_dataframe()
    test_orig_y = test_orig.labels.ravel()
    test_orig_x = test_orig_x[columns]

    c_charge_desc_cols = [c for c in dataset_orig.feature_names
                                    if c.startswith('c_charge_desc')]
    """
    bern_col = ['age_cat=25 - 45', 'age_cat=Greater than 45',
                'age_cat=Less than 25',
                'c_charge_degree=F', 'c_charge_degree=M'] + c_charge_desc_cols
    bern_indices = [list(train_orig_x.columns).index(c) for c in bern_col]

    geom_col = ['juv_fel_count', 'juv_misd_count',
                'juv_other_count', 'priors_count']
    geom_indices = [list(train_orig_x.columns).index(c) for c in geom_col]
    """
    bern_col = ['age_cat=25 to 45', 'age_cat=Greater than 45',
                'age_cat=Less than 25',
                'c_charge_degree=F', 'c_charge_degree=M',
                'priors_count=0', 'priors_count=1 to 3',
                'priors_count=More than 3'] + c_charge_desc_cols
    bern_indices = [list(train_orig_x.columns).index(c) for c in bern_col]

    geom_col = []
    geom_indices = [list(train_orig_x.columns).index(c) for c in geom_col]

    params = {'alpha': 1, 'bern_indices': bern_indices,
              'geom_indices': geom_indices}

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

    print('Orig group-wise')
    print(*['{:.4f}'.format(i) for i in p_result], sep='\t')
    print(*['{:.4f}'.format(i) for i in u_result], sep='\t')
    print(*['{:.4f}'.format(i) for i in m_result], sep='\t')
    print_model_performances(mod, test_orig_fd, verbose=True)
    dataset_balanced = get_balanced_dataset(dataset_orig, sample_mode=2)
    fix_balanced_dataset(dataset_balanced)
    # get_base_rates(dataset_balanced)
    columns = ['mean_difference', 'disparate_impact',
               'Opt_Acc', 'Priv_Acc', 'Unpriv_Acc',
               'Priv_True_pos', 'Priv_False_pos',
               'Unpriv_True_pos', 'Unpriv_False_pos']
    results = pd.DataFrame(columns=columns)

    args.k_fold = 5

    for alpha in [0.25, 0.5, 0.75]:
        for i in [23]:
            fd = get_real_fd(dataset_balanced, alpha=alpha)
            fd_train, fd_test = fd.split([0.7], shuffle=True, seed=i)

            # get_base_rates(fd_test)

            # fd_train = random_flips(fd_train, n=150)

            train_x, train_y = fd_train.get_xy(keep_protected=False)

            pmod, p_result = get_groupwise_performance(
                fd_train, fd_test, MixedModelNB,
                privileged=True, params=params, pos_rate=False
            )

            umod, u_result = get_groupwise_performance(
                fd_train, fd_test, MixedModelNB,
                privileged=False, params=params, pos_rate=False
            )
            mod, m_result = get_groupwise_performance(
                fd_train, fd_test, MixedModelNB,
                privileged=None, params=params, pos_rate=False
            )
            print()
            print(*['{:.4f}'.format(i) for i in p_result], sep='\t')
            print(*['{:.4f}'.format(i) for i in u_result], sep='\t')
            print(*['{:.4f}'.format(i) for i in m_result], sep='\t')

            print_model_performances(mod, test_fd=fd_test, verbose=True)
            break