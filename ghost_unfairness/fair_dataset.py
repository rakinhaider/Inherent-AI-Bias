import numpy as np
import pandas as pd

from aif360.metrics import (
    BinaryLabelDatasetMetric,
    ClassificationMetric
)
from aif360.datasets import StandardDataset, StructuredDataset
from itertools import product

# TODO: The following mapping is only necessary for de_dummy operation.
# It is only used for generating all groups.
# Might consider removing it later.
default_mappings = {
    'label_maps': [{1.0: 'Yes', 0.0: 'No'}],
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
                                 {1.0: 'Caucasian', 0.0: 'Not Caucasian'}]
}


class FairDataset(StandardDataset):
    def __init__(self, n_unprivileged, n_features,
                 label_name='label', favorable_classes=[0],
                 protected_attribute_names=['sex', 'race'],
                 privileged_classes=[['Male'], ['Caucasian']],
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=None,
                 features_to_drop=[], na_values=[],
                 custom_preprocessing=None,
                 metadata=default_mappings,
                 random_state=None,
                 alpha=0.5, beta=1):
        """
            Args:
                n_unprivileged (int): Number of instances in un_privileged group.
                    FairDataset class creates a dataset where each
                    unprivileged group has n_unprivileged number of
                     instances and each privileged group has
                     beta * n_unprivileged number of instances.

                n_features (int): Number of features in the dataset except 
                    the protected attributes.
                
                See :obj:`StandardDataset` for a description of the arguments.
        """
        _validate_alpha_beta(alpha, beta, n_unprivileged)
        generator = _get_groups(protected_attribute_names,
                                metadata['protected_attribute_maps'])
        df = pd.DataFrame()

        # TODO: Use different mean and variance for each attribute
        mu_ps = {'Male': 3, 'Female': 0}
        sigma_ps = {'Male': 2, 'Female': 5}
        mu_ns = {'Male': 10, 'Female': 10}
        sigma_ns = {'Male': 2, 'Female': 5}

        if random_state:
            np.random.seed(random_state)

        for d in generator:
            if _is_privileged(d, protected_attribute_names,
                              privileged_classes):
                n_samples = int(beta * n_unprivileged)
            else:
                n_samples = n_unprivileged

            n_pos = int(n_samples * alpha)
            n_neg = n_samples - n_pos

            mu_p = mu_ps
            sigma_p = sigma_ps
            mu_n = mu_ns
            sigma_n = sigma_ns
            for _, p in enumerate(protected_attribute_names):
                mu_p = mu_p[d[p]]
                sigma_p = sigma_p[d[p]]
                mu_n = mu_n[d[p]]
                sigma_n = sigma_n[d[p]]

            features = sigma_p * np.random.randn(n_pos, n_features) + mu_p
            group_df_pos = pd.DataFrame(features)
            features = sigma_n * np.random.randn(n_neg, n_features) + mu_n
            group_df_neg = pd.DataFrame(features)
            for feature in protected_attribute_names:
                group_df_pos[feature] = d[feature]
                group_df_neg[feature] = d[feature]

            group_df_pos[label_name] = np.ones(n_pos)
            group_df_neg[label_name] = np.zeros(n_neg)
            df = pd.concat([df, group_df_pos, group_df_neg],
                           ignore_index=True)

        if random_state:
            np.random.seed(None)

        super(FairDataset, self).__init__(df=df, label_name=label_name,
                                          favorable_classes=favorable_classes,
                                          protected_attribute_names=protected_attribute_names,
                                          privileged_classes=privileged_classes,
                                          instance_weights_name=instance_weights_name,
                                          categorical_features=categorical_features,
                                          features_to_keep=features_to_keep,
                                          features_to_drop=features_to_drop, na_values=na_values,
                                          custom_preprocessing=custom_preprocessing, metadata=metadata)

    def get_xy(self, keep_protected=False):
        x, _ = self.convert_to_dataframe()
        drop_fields = self.label_names.copy()
        if not keep_protected:
            drop_fields += self.protected_attribute_names

        x = x.drop(columns=drop_fields)

        y = self.labels.ravel()
        return x, y

    def _filter(self, columns, values):
        df, _ = self.convert_to_dataframe()
        for i, column in enumerate(columns):
            selection = None
            for val in values[i]:
                if selection:
                    selection = selection & (df[column] == val)
                else:
                    selection = df[column] == val

            df = df[selection]

        indices = [self.instance_names.index(i) for i in df.index]
        return self.subset(indices)

    def get_privileged_group(self):
        return self._filter(self.protected_attribute_names,
                            self.privileged_protected_attributes)

    def get_unprivileged_group(self):
        fd_priv = self._filter(self.protected_attribute_names,
                               self.privileged_protected_attributes)

        selected_rows = []
        for i in self.instance_names:
            if i not in fd_priv.instance_names:
                index = self.instance_names.index(i)
                selected_rows.append(index)

        return self.subset(selected_rows)


def _validate_alpha_beta(alpha, beta, n_unprivileged):
    value_error = True
    if (int(beta * n_unprivileged) * alpha * 1.0).is_integer():
        if (n_unprivileged * alpha * 1.0).is_integer():
            value_error = False
    if value_error:
        raise ValueError("Number of positive or negative instances "
                         "must be integer. \nun_privileged * alpha "
                         "or int(beta * un_privileged) * alpha results "
                         "in floating values.")


def _get_groups(attributes, value_maps):

    combinations = list(product(*value_maps))
    for comb in combinations:
        group = dict(zip(attributes, comb))
        for key in group:
            index = attributes.index(key)
            val = value_maps[index][group[key]]
            group[key] = val

        yield group


def _is_privileged(group, protected_attributes,
                   privileged_classes):

    for i in range(len(protected_attributes)):
        key = protected_attributes[i]
        if group[key] not in privileged_classes[i]:
            return False

    return True