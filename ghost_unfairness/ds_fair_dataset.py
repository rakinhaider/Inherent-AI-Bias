import numpy as np
import pandas as pd

from aif360.metrics import (
    BinaryLabelDatasetMetric,
    ClassificationMetric
)
from aif360.datasets import StandardDataset, StructuredDataset
from itertools import product
from ghost_unfairness.fair_dataset import FairDataset
from ghost_unfairness.fair_dataset import (
    default_mappings, _validate_alpha_beta, _is_privileged, _get_groups
)


class DSFairDataset(FairDataset):
    def __init__(self, n_unprivileged, n_features, n_redlin,
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
                 alpha=0.5, beta=1,
                 dist=None, verbose=False,
                 shift_random=0,
                 shift_priv=None):
        """
            Args:
                n_unprivileged (int): Number of instances in un_privileged group.
                    FairDataset class creates a dataset where each
                    unprivileged group has n_unprivileged number of
                    positive and negative instances and
                    each privileged group has beta * n_unprivileged
                    positive and negative instances.

                n_features (int): Number of features in the dataset except
                    the protected attributes.

                See :obj:`StandardDataset` for a description of the arguments.
        """

        _validate_alpha_beta(alpha, beta, n_unprivileged)
        generator = _get_groups(protected_attribute_names,
                                metadata['protected_attribute_maps'])
        df = pd.DataFrame()

        # TODO: Use different mean and variance for
        #  each protected attribute
        if not dist:
            mu_ps = {'p': 13, 'u': 10}
            sigma_ps = {'p': 2, 'u': 5}

            mu_ns = {'p': 3, 'u': 0}
            sigma_ns = {'p': 2, 'u': 5}
        else:
            mu_ps = dist['mu_ps']
            sigma_ps = dist['sigma_ps']

            mu_ns = dist['mu_ns']
            sigma_ns = dist['sigma_ns']

        if verbose:
            print(mu_ps)
            print(sigma_ps)
            print(mu_ns)
            print(sigma_ns)

        np.random.seed(47)

        # Feature means and variances.
        feat_mu_ps = [np.random.randint(0, 10)] * n_features
        feat_sigma_ps = [np.random.randint(1, 10)] * n_features
        # feat_mu_ns = [np.random.randint(0, 10)] * n_features
        # feat_sigma_ns = [np.random.randint(1, 10)] * n_features
        feat_mu_ns = feat_mu_ps
        feat_sigma_ns = feat_sigma_ps

        if random_state:
            np.random.seed(random_state)

        if verbose:
            print(feat_mu_ps)
            print(feat_mu_ns)
            print(feat_sigma_ps)
            print(feat_sigma_ns)

        for d in generator:
            if _is_privileged(d, protected_attribute_names,
                              privileged_classes):
                n_samples = int(beta * n_unprivileged)
                is_priv = True
            else:
                n_samples = n_unprivileged
                is_priv = False

            n_pos = int(n_samples * alpha)
            n_neg = n_samples - n_pos

            mu_p_p = mu_ps['p']
            sigma_p_p = sigma_ps['p']
            mu_p_n = mu_ns['p']
            sigma_p_n = sigma_ns['p']

            mu_u_p = mu_ps['u']
            sigma_u_p = sigma_ps['u']
            mu_u_n = mu_ns['u']
            sigma_u_n = sigma_ns['u']

            if is_priv:
                features = sigma_p_p * np.random.randn(n_pos, n_redlin) + mu_p_p
                m = (mu_u_p + mu_u_n) / 2
                column = sigma_u_p * np.random.randn(n_pos, n_redlin) + m
                if shift_priv:
                    column += shift_random
                features = np.hstack((features, column))
            else:
                features = sigma_u_p * np.random.randn(n_pos, n_redlin) + mu_u_p
                m = (mu_p_p + mu_p_n) / 2
                column = sigma_p_p * np.random.randn(n_pos, n_redlin) + m
                if shift_priv == False:
                    column += shift_random
                features = np.hstack((column, features))

            for i in range(n_features):
                rand = np.random.randn(n_pos, 1)
                column = feat_sigma_ps[i] * rand + feat_mu_ps[i]
                features = np.hstack((features, column))
            group_df_pos = pd.DataFrame(features)

            if is_priv:
                features = sigma_p_n * np.random.randn(n_neg, n_redlin) + mu_p_n
                m = (mu_u_p + mu_u_n) / 2
                column = sigma_u_p * np.random.randn(n_neg, n_redlin) + m
                if shift_priv:
                    column += shift_random
                features = np.hstack((features, column))
            else:
                features = sigma_u_n * np.random.randn(n_neg, n_redlin) + mu_u_n
                m = (mu_p_p + mu_p_n) / 2
                column = sigma_p_p * np.random.randn(n_neg, n_redlin) + m
                if shift_priv == False:
                    column += shift_random
                features = np.hstack((column, features))

            for i in range(n_features):
                rand = np.random.randn(n_neg, 1)
                column = feat_sigma_ns[i] * rand + feat_mu_ns[i]
                features = np.hstack((features, column))
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
        redline_cols = ['_'.join(['r', j, str(i)])for j in ['p', 'u']
                            for i in range(n_redlin)
                        ]
        feature_cols = ['f_' + str(i) for i in range(n_features)]
        columns = redline_cols + feature_cols
        columns += protected_attribute_names + [label_name]
        df.columns = columns

        super(FairDataset, self).__init__(
            df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata
        )