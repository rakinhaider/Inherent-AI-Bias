from oversampled_fair_dataset import get_dataset, get_group_dicts
from ghost_unfairness.sampling.synthetic_generator import group_indices, synthetic
from aif360.metrics import BinaryLabelDatasetMetric as BM
import numpy as np
from imblearn.over_sampling import ADASYN
import random


if __name__ == "__main__":
    data = get_dataset('compas')
    p_group, u_group = get_group_dicts(data)
    print(u_group)
    # print(p_group, u_group)
    unpriv_indices, priv_indices = group_indices(data, u_group)
    # print(un_priv_indices, priv_indices)
    # print(len(unpriv_indices), len(priv_indices))
    f_label = data.favorable_label
    uf_label = data.unfavorable_label
    p_group, u_group = get_group_dicts(data)

    dataset_metric = BM(data, u_group, p_group)

    base_rate_p = dataset_metric.base_rate(privileged=True)
    base_rate_u = dataset_metric.base_rate(privileged=False)
    print(base_rate_u, base_rate_p)
    data = data.subset(range(0, 20))
    dataset_balanced = synthetic(data, u_group, base_rate_p, base_rate_u,
                                 f_label, uf_label, None, 2, 1.00)
    # print(dataset_balanced.features)

    indices, priv_indices = group_indices(data, u_group)
    # print(len(indices), len(priv_indices))
    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset
    unprivileged_dataset = data.subset(indices)  # unprivileaged
    privileged_dataset = data.subset(priv_indices)  # privilegaed
    # print(privileged_dataset.features)
    # print(unprivileged_dataset.features)
    n_unpriv_favor = np.count_nonzero(
        unprivileged_dataset.labels == f_label)  # unprivileged with favorable label
    n_unpriv_unfavor = np.count_nonzero(
        unprivileged_dataset.labels != f_label)  # unprivileged with unfavorable label
    print(n_unpriv_unfavor, n_unpriv_favor)
    n_extra_sample = (base_rate_p * len(indices) - n_unpriv_favor) / (
                1 - base_rate_p) * 1.0

    print(n_extra_sample)

    # compute the ratio of dataset expansion for oversampling
    if n_extra_sample + n_unpriv_favor >= n_unpriv_unfavor:
        inflate_rate = int(
            ((n_extra_sample + n_unpriv_favor) / n_unpriv_unfavor) + 1)
    else:
        inflate_rate = round(
            ((n_extra_sample + n_unpriv_favor) / n_unpriv_unfavor) + 1)

    print(inflate_rate)

    data = unprivileged_dataset
    dataset_transf_train = data.copy(deepcopy=True)
    print(data.features.shape)
    f_dataset = data.subset(np.where(data.labels==f_label)[0].tolist())
    uf_dataset = data.subset(np.where(data.labels==uf_label)[0].tolist())
    print(f_dataset)
    print(uf_dataset)

    # expand the group with uf_label for oversampling purpose
    inflated_uf_features = np.repeat(uf_dataset.features, inflate_rate, axis=0)
    sample_features = np.concatenate((f_dataset.features, inflated_uf_features))
    print(sample_features.shape)
    inflated_uf_labels = np.repeat(uf_dataset.labels, inflate_rate, axis=0)
    sample_labels = np.concatenate((f_dataset.labels, inflated_uf_labels))

    oversample = ADASYN(sampling_strategy='minority')
    X, y = oversample.fit_resample(sample_features, sample_labels)
    y = y.reshape(-1, 1)

    print(len(X), len(y))

    # take samples from dataset with only favorable labels
    X = X[np.where(y==f_label)[0].tolist()]  # data with f_label + new samples
    y = y[y==f_label]
    print(len(X), len(y))

    selected = int(f_dataset.features.shape[0] + n_extra_sample)
    print(selected)

    X = X[:selected, :]
    y = y[:selected]
    y = y.reshape(-1,1)

    # set weights and protected_attributes for the newly generated samples
    inc = X.shape[0]-f_dataset.features.shape[0]
    new_weights = [random.choice(f_dataset.instance_weights) for _ in range(inc)]
    print(f_dataset.protected_attributes)
    new_attributes = [random.choice(f_dataset.protected_attributes) for _ in range(inc)]
    new_weights = np.array(new_weights)
    new_attributes = np.array(new_attributes)

    # compose transformed dataset
    dataset_transf_train.features = np.concatenate((uf_dataset.features, X))
    dataset_transf_train.labels = np.concatenate((uf_dataset.labels, y))
    dataset_transf_train.instance_weights = np.concatenate((uf_dataset.instance_weights, f_dataset.instance_weights, new_weights))
    dataset_transf_train.protected_attributes = np.concatenate((uf_dataset.protected_attributes, f_dataset.protected_attributes, new_attributes))

    # make a duplicate copy of the input data
    dataset_extra_train = data.copy()

    X_ex = X[-int(n_extra_sample):]
    y_ex = y[-int(n_extra_sample):]
    y_ex = y_ex.reshape(-1,1)

    # set weights and protected_attributes for the newly generated samples
    inc = int(n_extra_sample)
    new_weights = [random.choice(f_dataset.instance_weights) for _ in range(inc)]
    new_attributes = [random.choice(f_dataset.protected_attributes) for _ in range(inc)]

    # compose extra dataset
    dataset_extra_train.features = X_ex
    dataset_extra_train.labels = y_ex
    dataset_extra_train.instance_weights = np.array(new_weights)
    dataset_extra_train.protected_attributes = np.array(new_attributes)
