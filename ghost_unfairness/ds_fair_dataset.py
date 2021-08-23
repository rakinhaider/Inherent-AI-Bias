import numpy as np
import pandas as pd
from ghost_unfairness.fair_dataset import FairDataset
from ghost_unfairness.fair_dataset import (
    default_mappings, _validate_alpha_beta, _is_privileged, _get_groups
)


def add_features(config, n_feats, type, label, is_priv, n_rows):
    if label == 'pos':
        mu_key = 'mu_ps'
        sigma_key = 'sigma_ps'
    elif label == 'neg':
        mu_key = 'mu_ns'
        sigma_key = 'sigma_ns'

    features = np.array([]).reshape((n_rows, 0))
    if not config.get(type):
        return features

    mus = config[type][mu_key]['p' if is_priv else 'u']
    sigmas = config[type][sigma_key]['p' if is_priv else 'u']

    for i in range(0, n_feats):
        rand = np.random.randn(n_rows, 1)
        column = sigmas[i] * rand + mus[i]
        features = np.hstack((features, column))
    return features


class DSFairDataset(FairDataset):
    def __init__(self, n_unprivileged, n_indep_feat, n_redlin_feat,
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
                 shift_priv=None,
                 n_dep_feat=0):
        """
            Args:
                n_unprivileged (int): Number of instances in un_privileged group.
                    FairDataset class creates a dataset where each
                    unprivileged group has n_unprivileged number of
                    positive and negative instances and
                    each privileged group has beta * n_unprivileged
                    positive and negative instances.

                n_indep_feat (int): Number of features in the dataset except
                    the protected attributes.

                See :obj:`StandardDataset` for a description of the arguments.
        """

        _validate_alpha_beta(alpha, beta, n_unprivileged)
        generator = _get_groups(protected_attribute_names,
                                metadata['protected_attribute_maps'])
        df = pd.DataFrame()

        config = {}
        # TODO: Use different mean and variance for
        #  each protected attribute
        if not dist:
            config['redlin'] = {
                'mu_ps': {'p': 13, 'u': 10},
                'sigma_ps': {'p': 2, 'u': 5},
                'mu_ns': {'p': 3, 'u': 0},
                'sigma_ns': {'p': 2, 'u': 5}
            }
        else:
            config['redlin'] = dist

        if verbose:
            print(config)

        np.random.seed(47)

        # Feature means and variances.

        indep_feat_mu_ps = np.random.randint(0, 10, n_indep_feat)
        indep_feat_sigma_ps = np.random.randint(1, 10, n_indep_feat)
        config['indep'] = {
            'mu_ps': {'p': indep_feat_mu_ps, 'u': indep_feat_mu_ps},
            'sigma_ps': {'p': indep_feat_sigma_ps, 'u': indep_feat_sigma_ps},
            'mu_ns': {'p': indep_feat_mu_ps, 'u': indep_feat_mu_ps},
            'sigma_ns': {'p': indep_feat_sigma_ps, 'u': indep_feat_sigma_ps}
        }
        if verbose:
            print('{{\'indep\': {}}}'.format(config['indep']))

        if n_dep_feat:
            mu_ps = np.random.randint(0, 10, n_dep_feat)
            mu_ns = mu_ps - 5
            config['dep'] = {
                'mu_ps': {'p': mu_ps, 'u': mu_ps},
                'mu_ns': {'p': mu_ns, 'u': mu_ns}
            }
            sigmas = np.random.randint(1, 10, n_dep_feat)
            config['dep']['sigma_ps'] = {'p': sigmas, 'u': sigmas}
            config['dep']['sigma_ns'] = {'p': sigmas, 'u': sigmas}

            if verbose:
                print('{{\'dep\': {}}}'.format(config['dep']))

        if random_state:
            np.random.seed(random_state)

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

            mu_p_p = config['redlin']['mu_ps']['p']
            sigma_p_p = config['redlin']['sigma_ps']['p']
            mu_p_n = config['redlin']['mu_ns']['p']
            sigma_p_n = config['redlin']['sigma_ns']['p']

            mu_u_p = config['redlin']['mu_ps']['u']
            sigma_u_p = config['redlin']['sigma_ps']['u']
            mu_u_n = config['redlin']['mu_ns']['u']
            sigma_u_n = config['redlin']['sigma_ns']['u']

            if is_priv:
                features = sigma_p_p * np.random.randn(n_pos, n_redlin_feat) \
                           + mu_p_p
                m = (mu_u_p + mu_u_n) / 2
                column = sigma_u_p * np.random.randn(n_pos, n_redlin_feat) + m
                if shift_priv:
                    column += shift_random
                features = np.hstack((features, column))
            else:
                features = sigma_u_p * np.random.randn(n_pos, n_redlin_feat) \
                           + mu_u_p
                m = (mu_p_p + mu_p_n) / 2
                column = sigma_p_p * np.random.randn(n_pos, n_redlin_feat) + m
                if shift_priv == False:
                    column += shift_random
                features = np.hstack((column, features))

            cols = add_features(config, n_indep_feat,
                                'indep', 'pos', is_priv, n_pos)
            features = np.hstack((features, cols))

            cols = add_features(config, n_dep_feat,
                                'dep', 'pos', is_priv, n_pos)
            features = np.hstack((features, cols))

            group_df_pos = pd.DataFrame(features)

            if is_priv:
                features = sigma_p_n * np.random.randn(n_neg,
                                                       n_redlin_feat) + mu_p_n
                m = (mu_u_p + mu_u_n) / 2
                column = sigma_u_p * np.random.randn(n_neg, n_redlin_feat) + m
                if shift_priv:
                    column += shift_random
                features = np.hstack((features, column))
            else:
                features = sigma_u_n * np.random.randn(n_neg,
                                                       n_redlin_feat) + mu_u_n
                m = (mu_p_p + mu_p_n) / 2
                column = sigma_p_p * np.random.randn(n_neg, n_redlin_feat) + m
                if shift_priv == False:
                    column += shift_random
                features = np.hstack((column, features))

            cols = add_features(config, n_indep_feat,
                                'indep', 'neg', is_priv, n_neg)
            features = np.hstack((features, cols))

            cols = add_features(config, n_dep_feat,
                                'dep', 'neg', is_priv, n_neg)
            features = np.hstack((features, cols))

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
        redline_cols = ['_'.join(['r', j, str(i)]) for j in ['p', 'u']
                        for i in range(n_redlin_feat)
                        ]
        indep_cols = ['i_' + str(i) for i in range(n_indep_feat)]
        dep_cols = ['d_' + str(i) for i in range(n_dep_feat)]
        columns = redline_cols + indep_cols + dep_cols
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
