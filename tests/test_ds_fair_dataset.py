from unittest import TestCase

import numpy as np
import pandas as pd
from itertools import product

from aif360.metrics import BinaryLabelDatasetMetric
from inherent_bias.ds_fair_dataset import DSFairDataset
from inherent_bias.fair_dataset import default_mappings

class TestDSFairDataset(TestCase):
    """Tests for Guttman R-Tree implementation"""
    protected = ['sex']
    privileged_classes = [['Male']]

    privileged_group = [{'sex': 1}]
    unprivileged_group = [{'sex': 0}]

    mapping = [{1.0: 'Male', 0.0: 'Female'}]

    def test_dsfair(self):
        np.random.seed(23)
        beta = 1
        n_unprivileged = 100000
        alpha = 0.5
        metadata = default_mappings.copy()
        metadata['protected_attribute_maps'] = self.mapping
        fd = DSFairDataset(n_unprivileged, 2, 1,
                           protected_attribute_names=self.protected,
                           privileged_classes=self.privileged_classes,
                           beta=beta,
                           alpha=alpha,
                           metadata=metadata,
                           shift_random=2,
                           shift_priv=None,
                           verbose=True,
                           n_dep_feat=1)

        df, _ = fd.convert_to_dataframe()

        groups = df.groupby(['sex', 'label'])

        for tup, g in groups:
            # print(g)
            print(g.describe())