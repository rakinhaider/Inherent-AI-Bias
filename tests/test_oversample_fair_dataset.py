import unittest
from oversampled_compas_experiment import *
from aif360.datasets import BinaryLabelDataset
import pandas as pd


class TestOversampledFD(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.df = pd.DataFrame([[1, 2, 0, 1],
                                [1, 2, 1, 1],
                                [3, 4, 1, 0],
                                [5, 6, 0, 0],
                                [7, 8, 1, 1]],
                               columns=[0, 1, 'sex', 'labels'])
        self.bld = BinaryLabelDataset(1, 0, df=self.df,
                                      label_names=['labels'],
                                      protected_attribute_names=['sex'])
        super().__init__(*args, **kwargs)

    def test_get_group_dicts(self):
        priv, unpriv = get_group_dicts(self.bld)
        assert priv == [{'sex': 1}]
        assert unpriv == [{'sex': 0}]

    def test_get_samples_by_group(self):
        names = get_samples_by_group(self.bld, 2, [{'sex': 1}],
                                     self.bld.favorable_label)
        names.sort()
        assert all(names == [1, 4])

    def test_get_indices(self):
        priv, unpriv = get_group_dicts(self.bld)
        indices = get_indices(self.bld, condition=priv, label=1)
        indices.sort()
        assert indices == [1, 4]

        indices = get_indices(self.bld, condition=unpriv, label=1)
        indices.sort()
        assert indices == [0]

        indices = get_indices(self.bld, condition=priv, label=0)
        indices.sort()
        assert indices == [2]

        indices = get_indices(self.bld, condition=unpriv, label=0)
        indices.sort()
        assert indices == [3]

        indices = get_indices(self.bld, condition=None, label=0)
        indices.sort()
        assert indices == [2, 3]

        indices = get_indices(self.bld, condition=priv, label=None)
        indices.sort()
        assert indices == [1, 2, 4]