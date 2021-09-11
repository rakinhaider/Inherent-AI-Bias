import numpy as np
from metrics_utils import compute_metrics, describe_metrics, get_test_metrics, test

#setup test models
from models import TModel

# Metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics import utils

#Bias mitigation techniques
from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR, OptimPreproc, Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing, ARTClassifier, GerryFairClassifier, MetaFairClassifier, PrejudiceRemover
from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing\
        import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing\
        import EqOddsPostprocessing
#from aif360.algorithms.postprocessing.reject_option_classification\
#        import RejectOptionClassification
from aif360.algorithms.postprocessing import RejectOptionClassification

# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

#oversampling
from oversample import synthetic


class BaseMitigator:

    def __init__(self):
        pass

    def run_mitigator(self):
        pass


# no mitigator
class NullMitigator(BaseMitigator):

    mitigator_type = 'No Mitigator'
    def run_mitigator(self, dataset_orig_train, dataset_orig_val, dataset_orig_test,
                      model_type, orig_metrics,
                      f_label, uf_label,
                      unprivileged_groups, privileged_groups,
                      THRESH_ARR, DISPLAY, SCALER):
        dataset = dataset_orig_train
        metrics = get_test_metrics(dataset, dataset_orig_val, dataset_orig_test, model_type, orig_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)

        return metrics


class SyntheticMitigator(BaseMitigator):

    mitigator_type = 'Synthetic Data Mitigator'
    def run_mitigator (self, dataset_orig_train, dataset_orig_val, dataset_orig_test,
                       privileged_groups, unprivileged_groups,
                       base_rate_privileged, base_rate_unprivileged,
                       model_type, transf_metrics,
                       f_label, uf_label,
                       THRESH_ARR, DISPLAY, OS_MODE, SCALER):
        # generating synthetic data
        dataset_transf_train = synthetic(dataset_orig_train, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, f_label, uf_label, os_mode = OS_MODE)
        # print('origin, transf: ', dataset_orig_train.features.shape[0], dataset_transf_train.features.shape[0])

        metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
        # print('after transf priv: ', metric_transf_train.base_rate(privileged=True))
        # print('after transf unpriv: ', metric_transf_train.base_rate(privileged=False))
        # print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())


        # fitting the model on the transformed dataset with synthetic generator
        dataset = dataset_transf_train
        transf_metrics = get_test_metrics(dataset, dataset_orig_val, dataset_orig_test, model_type, transf_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)

        return metric_transf_train, transf_metrics

