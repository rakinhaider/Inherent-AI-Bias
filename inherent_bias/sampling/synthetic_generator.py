# Collected from Yan Zhou

import sys
import random
import numpy as np
from aif360.metrics import utils
from inherent_bias.sampling.myutils import balance

# return dataset indices of unprivileged and privileaged groups
def group_indices(dataset, unprivileged_groups):
    feature_names = dataset.feature_names
    cond_vec = utils.compute_boolean_conditioning_vector(dataset.features, feature_names, unprivileged_groups)

    # indices of examples in the unprivileged and privileged groups
    indices = [i for i, x in enumerate(cond_vec) if x == True]
    priv_indices = [i for i, x in enumerate(cond_vec) if x == False]
    return indices, priv_indices

# oversample unfavorable in the privileged group
def synthetic_unfavor_priv(dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00):

    indices, priv_indices = group_indices(dataset, unprivileged_groups)

    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset 
    unprivileged_dataset = dataset.subset(indices) # unprivileaged
    privileged_dataset = dataset.subset(priv_indices) # privilegaed

    n_priv_favor = np.count_nonzero(privileged_dataset.labels==f_label) # privileged with favorable label
    n_priv_unfavor = np.count_nonzero(privileged_dataset.labels!=f_label) # privileged with unfavorable label

    n_extra_sample = (n_priv_favor - bnp * len(priv_indices)) / bnp * sampling_strategy

    # compute the ratio of dataset expansion for oversampling  
    if n_extra_sample + n_priv_unfavor >= n_priv_favor:
        inflate_rate = int(((n_extra_sample+n_priv_unfavor)/n_priv_favor)+1)
    else:
        inflate_rate = round(((n_extra_sample+n_priv_unfavor)/n_priv_favor)+1)

    dataset_transf_refprivileged_train, extra_unfavored_priv  = balance(privileged_dataset, n_extra_sample, inflate_rate, uf_label, f_label)

    return dataset_transf_refprivileged_train, extra_unfavored_priv


# oversample favorable in the unprivileged group
def synthetic_favor_unpriv (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00):

    indices, priv_indices = group_indices (dataset, unprivileged_groups)

    # subset: unprivileged--unprivileged_dataset and privileged--privileged_dataset 
    unprivileged_dataset = dataset.subset(indices) # unprivileaged
    privileged_dataset = dataset.subset(priv_indices) # privilegaed

    n_unpriv_favor = np.count_nonzero(unprivileged_dataset.labels==f_label) # unprivileged with favorable label
    n_unpriv_unfavor = np.count_nonzero(unprivileged_dataset.labels!=f_label) # unprivileged with unfavorable label

    n_extra_sample = (bp * len(indices)-n_unpriv_favor) / (1- bp) * sampling_strategy

    # compute the ratio of dataset expansion for oversampling  
    if n_extra_sample + n_unpriv_favor >= n_unpriv_unfavor:
        inflate_rate = int(((n_extra_sample+n_unpriv_favor)/n_unpriv_unfavor)+1)
    else:
        inflate_rate = round(((n_extra_sample+n_unpriv_favor)/n_unpriv_unfavor)+1)

    dataset_transf_refprivileged_train, extra_favored_unpriv  = balance(unprivileged_dataset, n_extra_sample, inflate_rate, f_label, uf_label)

    return dataset_transf_refprivileged_train, extra_favored_unpriv


# adaptive oversampling for the unprivileged group
def synthetic(dataset, unprivileged_groups,
              bp, bnp, f_label, uf_label,
              model_type=None, os_mode=2,
              sampling_strategy=0.50):

    # make a duplicate copy of the input data
    dataset_transf_train = dataset.copy(deepcopy=True)
    # case: if privileged is favored, i.e. has a higher base rate

    # [Method 1] inflate privileged unfavored class
    if os_mode == 1:
        _, sample_unfavor_priv = synthetic_unfavor_priv (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00)
        dataset_transf_train.features = np.concatenate((dataset_transf_train.features, sample_unfavor_priv.features))
        dataset_transf_train.labels = np.concatenate((dataset_transf_train.labels, sample_unfavor_priv.labels))
        dataset_transf_train.instance_weights = np.concatenate((dataset_transf_train.instance_weights, sample_unfavor_priv.instance_weights))
        dataset_transf_train.protected_attributes = np.concatenate((dataset_transf_train.protected_attributes, sample_unfavor_priv.protected_attributes))
    elif os_mode == 2:
    # [Method 2] inflate unprivileged favored class
        _, sample_favor_unpriv = synthetic_favor_unpriv (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00)
        dataset_transf_train.features = np.concatenate((dataset_transf_train.features, sample_favor_unpriv.features))
        dataset_transf_train.labels = np.concatenate((dataset_transf_train.labels, sample_favor_unpriv.labels))
        dataset_transf_train.instance_weights = np.concatenate((dataset_transf_train.instance_weights, sample_favor_unpriv.instance_weights))
        dataset_transf_train.protected_attributes = np.concatenate((dataset_transf_train.protected_attributes, sample_favor_unpriv.protected_attributes))
    # [Method 3] combine methods 1 and 2
    elif os_mode == 3:
        _, sample_unfavor_priv = synthetic_unfavor_priv (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00)
        dataset_transf_train.features = np.concatenate((dataset_transf_train.features, sample_unfavor_priv.features))
        dataset_transf_train.labels = np.concatenate((dataset_transf_train.labels, sample_unfavor_priv.labels))
        dataset_transf_train.instance_weights = np.concatenate((dataset_transf_train.instance_weights, sample_unfavor_priv.instance_weights))
        dataset_transf_train.protected_attributes = np.concatenate((dataset_transf_train.protected_attributes, sample_unfavor_priv.protected_attributes))

        _, sample_favor_unpriv = synthetic_favor_unpriv (dataset, unprivileged_groups, bp, bnp, f_label, uf_label, sampling_strategy=1.00)
        dataset_transf_train.features = np.concatenate((dataset_transf_train.features, sample_favor_unpriv.features))
        dataset_transf_train.labels = np.concatenate((dataset_transf_train.labels, sample_favor_unpriv.labels))
        dataset_transf_train.instance_weights = np.concatenate((dataset_transf_train.instance_weights, sample_favor_unpriv.instance_weights))
        dataset_transf_train.protected_attributes = np.concatenate((dataset_transf_train.protected_attributes, sample_favor_unpriv.protected_attributes))
    else:
        sys.exit("Oversampling mode is missing: 1: oversample unfavorable privileged; 2: oversample favorable unprivileged; 3. both")

    # dataset_transf_train.instance_names = np.concatenate(dataset_transf_train.instance_names,)

    return dataset_transf_train

