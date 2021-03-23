import unittest
from ghost_unfairness.oversampled_fair_dataset import *
from aif360.datasets import BinaryLabelDataset
import pandas as pd
import numpy as np

class TestOversampledFD(unittest.TestCase):
    def test_get_samples_by_group(self):
        df = pd.DataFrame([[1, 2, 0, 1],
                           [1, 2, 1, 1],
                           [3, 4, 1, 0],
                           [5, 6, 0, 0],
                           [7, 8, 1, 1]],
                          columns=[0, 1, 'sex', 'labels'])
        bld = BinaryLabelDataset(1, 0, df=df,
                                 label_names=['labels'],
                                 protected_attribute_names=['sex'])
        names = get_samples_by_group(bld, 2, [{'sex': 1}], True)
        names.sort()
        assert names == ['1', '4']
