from ghost_unfairness.sampling.synthetic_generator import(
    synthetic, synthetic_favor_unpriv, synthetic_unfavor_priv,
    group_indices
)
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric as BM
import numpy as np
from sklearn.naive_bayes import GaussianNB
from aif360.metrics.utils import(
    compute_boolean_conditioning_vector as condition_vec,
    compute_num_pos_neg
)
from ghost_unfairness.utils import get_groupwise_performance
from ghost_unfairness.fair_dataset import FairDataset


def get_group_dicts(dataset):
    if len(dataset.protected_attribute_names):
        p_attr = dataset.protected_attribute_names[0]
        return [{p_attr: 1}], [{p_attr: 0}]
    else:
        raise NotImplementedError("Not implemented.")


def get_indices_by_label(dataset, is_favor=True):
    if is_favor:
        label = dataset.favorable_label
    else:
        label = dataset.unfavorable_label
    condition_vector = (dataset.labels == label)
    return [i for i in range(len(condition_vector)) if condition_vector[i]]


def get_pos_neg_by_group(dataset, group, is_favor=True):
    if is_favor:
        label = dataset.favorable_label
    else:
        label = dataset.unfavorable_label
    return compute_num_pos_neg(dataset.features, dataset.labels,
                               dataset.instance_weights,
                               dataset.feature_names,
                               label, group)


def get_samples_by_group(dataset, n_samples, group, is_favor=True):
    cond_vec = condition_vec(dataset.features, dataset.feature_names,
                             group)
    indices = [i for i, x in enumerate(cond_vec) if x]
    subset = dataset.subset(indices)
    indices = get_indices_by_label(subset, is_favor=is_favor)
    indices = np.random.choice(indices, n_samples, replace=False)
    return [subset.instance_names[i] for i in indices]


def get_fd_from_balanced(dataset, alpha=0.5, beta=1):
    # First find n_unprivileged
    # Then check number of samples to take from each group
    # Then sample and merge them together.
    p_group, u_group = get_group_dicts(dataset)
    p_pos = get_pos_neg_by_group(dataset, p_group, True)
    p_neg = get_pos_neg_by_group(dataset, p_group, False)
    u_pos = get_pos_neg_by_group(dataset, u_group, True)
    u_neg = get_pos_neg_by_group(dataset, u_group, False)

    n_unpriv = min(u_pos/alpha, u_neg/(1 - alpha),
                   p_pos/(alpha * beta), p_neg/((1-alpha)*beta)
    )

    temp = [(alpha * n_unpriv, u_group, True),
            ((1 - alpha) * n_unpriv, u_group, False),
            (alpha * (beta * n_unpriv), p_group, True),
            ((1 - alpha) * (beta * n_unpriv), p_group, False)]
    instance_names = []
    for n, g, f in temp:
        print(n)
        sample_indices = get_samples_by_group(dataset, round(n), g, f)
        instance_names.extend(sample_indices)

    print(len(instance_names))
    indices = [dataset.instance_names.index(i) for i in instance_names]
    return dataset.subset(indices)


if __name__ == "__main__":

    dataset_orig = CompasDataset(
        protected_attribute_names=['race'],     # this dataset also contains protected
                                                # attribute for "sex" which we do not
                                                # consider in this evaluation
        privileged_classes=[['Caucasian']],     # race Caucasian is considered privileged
        features_to_drop=['personal_status', 'sex']  # ignore sex-related attributes
    )

    features = dataset_orig.features
    f_label = dataset_orig.favorable_label
    uf_label = dataset_orig.unfavorable_label
    p_group, u_group = get_group_dicts(dataset_orig)

    sens_attr = dataset_orig.protected_attribute_names[0]
    dataset_metric = BM(dataset_orig, u_group, p_group)

    base_rate_p = dataset_metric.base_rate(privileged=True)
    base_rate_u = dataset_metric.base_rate(privileged=False)
    dataset_balanced = synthetic(dataset_orig, u_group,
                                 base_rate_p, base_rate_u,
                                 f_label, uf_label,
                                 None, 2, 1.00)

    n = len(dataset_balanced.features)
    # Fixing balanced dataset attributes
    dataset_balanced.instance_names = [str(i) for i in range(n)]
    dataset_balanced.scores = np.ones_like(dataset_balanced.labels)
    dataset_balanced.scores -= dataset_balanced.labels

    np.random.seed(47)
    fair_real_dataset = get_fd_from_balanced(dataset_balanced)
    # fair_real_dataset = get_fd_from_balanced(dataset_balanced)
    dataset_metric = BM(fair_real_dataset, u_group, p_group)
    print(dataset_metric.base_rate(privileged=True))
    print(dataset_metric.base_rate(privileged=False))
    print(dataset_metric.base_rate(privileged=None))

    fd = FairDataset(0, 0, 0)
    fd.update_from_dataset(fair_real_dataset)
    pmod, p_result = get_groupwise_performance(fd, fd, GaussianNB,
                                               privileged=True)
    print(*p_result, sep='\t')
    umod, u_result = get_groupwise_performance(fd, fd, GaussianNB,
                                               privileged=False)
    print(*u_result, sep='\t')
    mod, result = get_groupwise_performance(fd, fd, GaussianNB,
                                               privileged=None)
    print(*result, sep='\t')
