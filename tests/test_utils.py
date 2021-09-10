import unittest
import numpy as np
from inherent_bias.fair_dataset import FairDataset
from aif360.metrics import ClassificationMetric
from inherent_bias.utils import (
    get_positive_rate, get_single_prot_default_map
)


class TestUtils(unittest.TestCase):
    def test_get_delta(self):
        protected = ["sex"]
        privileged_classes = [['Male']]

        kwargs = {'protected_attribute_names': protected,
                  'privileged_classes': privileged_classes,
                  'favorable_classes': [1],
                  'label_name': 'labels',
                  'metadata': get_single_prot_default_map()
                  }
        fd = FairDataset(8, 0, 1, **kwargs)
        fd_pred = fd.copy(deepcopy=True)
        np.random.seed(123)
        fd_pred.labels = np.random.randint(0, 2, 16)
        cm = ClassificationMetric(fd, fd_pred,
                                  fd.unprivileged_groups,
                                  fd.privileged_groups)
        assert get_positive_rate(cm, True, True) == 1/4
        assert get_positive_rate(cm, True, False) == 1/4
        assert get_positive_rate(cm, False, True) == 3/4
        assert get_positive_rate(cm, False, False) == 1/2
        assert get_positive_rate(cm, None, True) == 1/2
        assert get_positive_rate(cm, None, False) == 3/8



