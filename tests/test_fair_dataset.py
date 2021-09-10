from unittest import TestCase

import numpy as np
import pandas as pd
from itertools import product

from aif360.metrics import BinaryLabelDatasetMetric
from inherent_bias.fair_dataset import FairDataset
from inherent_bias.fair_dataset import(
    _get_groups, _is_privileged, default_mappings
)


class TestFairDataset(TestCase):
    """Tests for Guttman R-Tree implementation"""
    protected = ['sex', 'race']
    privileged_classes = [['Male'], ['Caucasian']]

    privileged_group = [{'sex': 1, 'race': 1}]
    unprivileged_group = [{'sex': 0}, {'sex': 1, 'race': 0}]

    mapping = [{1.0: 'Male', 0.0: 'Female'},
               {1.0: 'Caucasian', 0.0: 'Not Caucasian'}]

    def test_non_redlining(self):
        np.random.seed(23)
        beta = 1
        n_unprivileged = 10000
        alpha = 0.5
        metadata = default_mappings.copy()
        metadata['protected_attribute_maps'] = self.mapping
        fd = FairDataset(n_unprivileged, 3, 1,
                         protected_attribute_names=self.protected,
                         privileged_classes=self.privileged_classes,
                         beta=beta,
                         alpha=alpha,
                         metadata=metadata)

        df, _ = fd.convert_to_dataframe()
        pd.set_option('display.max_columns', None)
        print(df[df['label'] == 1].describe())
        print(df[df['label'] == 0].describe())

    def test_init(self):
        betas = [0.5, 1, 2]
        n_unprivileged = 10
        alphas = [0, 0.25, 0.5, 0.75, 1]
        for alpha, beta in product(alphas, betas):
            # print(alpha, beta, n_unprivileged)
            try:
                self._test_init(alpha, beta, n_unprivileged)
            except ValueError:
                assert True

    def _test_init(self, alpha, beta, n_unprivileged):
        metadata = default_mappings.copy()
        metadata['protected_attribute_maps'] = self.mapping
        fd = FairDataset(n_unprivileged, 3, 1,
                         protected_attribute_names=self.protected,
                         privileged_classes=self.privileged_classes,
                         beta=beta,
                         alpha=alpha,
                         metadata=metadata)

        df, _ = fd.convert_to_dataframe()
        groups, counts = np.unique(df[self.protected],
                                   axis=0,
                                   return_counts=True)
        # print(counts)
        # Test correct number of instance in each group.
        for i in range(len(groups)):
            group = {}
            for j in range(len(self.protected)):
                p = self.protected[j]
                group[p] = self.mapping[j][groups[i][j]]

            if _is_privileged(group, self.protected,
                              self.privileged_classes):
                assert np.ceil(beta * n_unprivileged) == counts[i]
            else:
                assert n_unprivileged == counts[i]

        # Test dataset fairness metrics.

        c_metric = BinaryLabelDatasetMetric(fd,
                        unprivileged_groups=self.unprivileged_group,
                        privileged_groups=self.privileged_group)

        if alpha == 1:
            assert np.isnan(c_metric.disparate_impact())
        else:
            assert c_metric.disparate_impact() == 1
        assert c_metric.mean_difference() == 0

    def test_get_privileged_group(self):
        # Protected attribute for privileged classes
        # are always mapped to 1.0
        # irrespective of the default_labelling.
        # If you need to access them using class names i.e. "Male"
        # you will need to de_dummy the dataframe.
        protected_classes_values = [[1], [1]]

        fd = FairDataset(2, 2, 1,
                         protected_attribute_names=self.protected,
                         privileged_classes=self.privileged_classes,
                         )

        fd_priv = fd.get_privileged_group()
        df, attr = fd.convert_to_dataframe()
        selection = None
        for i, p_attr in enumerate(self.protected):
            for pc_val in protected_classes_values[i]:
                if selection is not None:
                    selection = selection & (df[p_attr] == pc_val)
                else:
                    selection = df[p_attr] == pc_val

        df = df[selection]
        for i in df.index:
            assert i in fd_priv.instance_names
            # TODO: Match feature values for the index

        assert (len(df)) == len(fd_priv.instance_names)

        # TODO: Match types of each attribute

    def test_get_unprivileged_group(self):
        fd = FairDataset(2, 2, 1,
                         protected_attribute_names=self.protected,
                         privileged_classes=self.privileged_classes,
                         )

        fd_unpriv = fd.get_unprivileged_group()
        # print(fd_unpriv.convert_to_dataframe())

        # TODO: Check types of each attribute
        # TODO: Check the feature values of each
        #  instance matches the original

    def test_get_groups(self):
        mapping = [{1.0: 'Male', 0.0: 'Female'},
                   {1.0: 'Caucasian', 0.0: 'Not Caucasian'}]

        groups = _get_groups(['sex', 'race'],
                             mapping)

        group_results = [{'sex': 'Female', 'race': 'Not Caucasian'},
                         {'sex': 'Female', 'race': 'Caucasian'},
                         {'sex': 'Male', 'race': 'Not Caucasian'},
                         {'sex': 'Male', 'race': 'Caucasian'}]

        for group in groups:
            assert group in group_results

    def test_is_privileged(self):
        boolean = _is_privileged({'sex': 'Female', 'race': 'Not Caucasian'},
                                 self.protected,
                                 self.privileged_classes)
        assert not boolean
        boolean = _is_privileged({'sex': 'Male', 'race': 'Caucasian'},
                                 self.protected,
                                 self.privileged_classes)
        assert boolean

        boolean = _is_privileged({'sex': 'Male', 'race': 'Caucasian'},
                                 self.protected,
                                 [['Male'], ['Caucasian', 'Not Caucasian']])
        assert boolean
        boolean = _is_privileged({'sex': 'Male', 'race': 'Not Caucasian'},
                                 self.protected,
                                 [['Male'], ['Caucasian', 'Not Caucasian']])
        assert boolean