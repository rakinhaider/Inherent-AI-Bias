import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import statistics
from data_utils import DatasetBuilder
from test_algorithms import TestAlgorithms
from plot_utils import plot_algo

# Metrics
from aif360.metrics import BinaryLabelDatasetMetric

# construct argument parser
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", default='compas', help="dataset: compas")
ap.add_argument("-c", "--classifier", default='nb', help="baseline model: nb")
ap.add_argument("-m", "--mitigator", required=False, help="mitigators: ")
ap.add_argument("-o", "--os", default=2, help="oversample mode: 1: privi unfav 2: unpriv fav")
args = vars(ap.parse_args())

DATASET = args["data"]
BASELINE = args["classifier"]
OS_MODE = int(args["os"])

# global constants
SCALER = False 
DISPLAY = False 
THRESH_ARR = 0.5

# loop ten times 
N = 10 
# percentage of favor and unfavor
priv_metric_orig = defaultdict(float)
favor_metric_orig = defaultdict(float)
favor_metric_transf = defaultdict(float)

# running resutls over N runs
orig_metrics = defaultdict(list)
transf_metrics = defaultdict(list)

# load dataset and set the groups
dataset_builder =  DatasetBuilder(DATASET)
dataset_orig = dataset_builder.load_data()
sens_attr = dataset_orig.protected_attribute_names[0]
unprivileged_groups = dataset_builder.unprivileged_groups
privileged_groups = dataset_builder.privileged_groups

# training data split ratio
p = 0.8

# run mitigating algorithms
for i in range(N):
    # split dataset into train, validation, and test
    dataset_orig_train, dataset_orig_test = dataset_orig.split([p], shuffle=True)
    dataset_orig_val = dataset_orig_test

    # favorable and unfavorable labels and feature_names
    f_label = dataset_orig_train.favorable_label
    uf_label = dataset_orig_train.unfavorable_label
    feature_names = dataset_orig_train.feature_names

    # show data info
    # print("#### Training Dataset shape")
    # print(dataset_orig_train.features.shape)
    # print("#### Favorable and unfavorable labels")
    # print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
    # print("#### Protected attribute names")
    # print(dataset_orig_train.protected_attribute_names)
    # print("#### Privileged and unprivileged protected attribute values")
    # print(privileged_groups, unprivileged_groups)
    # print("#### Dataset feature names")
    # print(dataset_orig_train.feature_names)
    # print(dataset_orig_train.features[0])

    # check fairness on the original data
    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    # print("privileged vs. unprivileged: ", metric_orig_train.num_positives(privileged=True) + metric_orig_train.num_negatives(privileged=True), metric_orig_train.num_positives(privileged=False) + metric_orig_train.num_negatives(privileged=False))
    base_rate_unprivileged = metric_orig_train.base_rate(privileged=False)
    base_rate_privileged = metric_orig_train.base_rate(privileged=True)
    # print('base_pos unpriv: ', base_rate_unprivileged)
    # print('base_pos priv: ', base_rate_privileged)
    #print(np.count_nonzero(dataset_orig_train.labels==f_label))
    # print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

    # statistics of favored/positive class BEFORE transf 
    priv_metric_orig['total_priv'] += metric_orig_train.num_instances(privileged = True) 
    priv_metric_orig['total_unpriv'] += metric_orig_train.num_instances(privileged = False) 
    favor_metric_orig['total_favor'] += metric_orig_train.base_rate()
    favor_metric_orig['total_unfavor'] += 1 - metric_orig_train.base_rate()
    favor_metric_orig['priv_favor'] += metric_orig_train.base_rate(privileged = True)
    favor_metric_orig['priv_unfavor'] += 1 - metric_orig_train.base_rate(privileged = True)
    favor_metric_orig['unpriv_favor'] += metric_orig_train.base_rate(privileged = False)
    favor_metric_orig['unpriv_unfavor'] += 1 - metric_orig_train.base_rate(privileged = False)

    # print(dataset_orig_train.features.shape, dataset_orig_val.features.shape, dataset_orig_test.features.shape)

    # testing mitigation methods 
    test_cases = TestAlgorithms(BASELINE)

    # null mitigator
    orig_metrics = test_cases.run_original(dataset_orig_train, dataset_orig_val, dataset_orig_test, BASELINE, orig_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER) 

    # synthetic data mitigator
    metric_transf_train, transf_metrics = test_cases.run_oversample(dataset_orig_train, dataset_orig_val, dataset_orig_test, privileged_groups, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, BASELINE, transf_metrics, f_label, uf_label, THRESH_ARR, DISPLAY, OS_MODE, SCALER)
    # statistics of favored/positive class AFTER transf
    favor_metric_transf['total_favor'] += metric_transf_train.base_rate()
    favor_metric_transf['total_unfavor'] += 1 - metric_transf_train.base_rate()
    favor_metric_transf['priv_favor'] += metric_transf_train.base_rate(privileged = True)
    favor_metric_transf['priv_unfavor'] += 1 - metric_transf_train.base_rate(privileged = True)
    favor_metric_transf['unpriv_favor'] += metric_transf_train.base_rate(privileged = False)
    favor_metric_transf['unpriv_unfavor'] += 1 - metric_transf_train.base_rate(privileged = False)


# display output

# print('\n\n\n')
# print(DATASET)
# print(dataset_orig_train.features.shape[0])
# print('\n\n\n')
priv_metric_orig = {k: [v/N] for (k,v) in priv_metric_orig.items()}
results = [priv_metric_orig]
tr = pd.Series(['orig'], name='num_instance')
df = pd.concat([pd.DataFrame(metrics) for metrics in results], axis = 0).set_index([tr])
# print(df)

# print('\n')
favor_metric_orig = {k: [v/N] for (k,v) in favor_metric_orig.items()}
favor_metric_transf = {k: [v/N] for (k,v) in favor_metric_transf.items()}
pd.set_option('display.multi_sparse', False)
results = [favor_metric_orig, favor_metric_transf]
tr = pd.Series(['orig'] + ['transf'], name='dataset')
df = pd.concat([pd.DataFrame(metrics) for metrics in results], axis = 0).set_index([tr])
# print(df)

# print('\n\n\n')

# dataframe to display fairness metrics
# error metrics
orig_error_metrics = {k: [statistics.stdev(v)] for (k,v) in orig_metrics.items()}
transf_error_metrics = {k: [statistics.stdev(v)] for (k,v) in transf_metrics.items()}

# mean value metrics
orig_metrics_mean = {k: [sum(v)/N] for (k,v) in orig_metrics.items()}
transf_metrics_mean = {k: [sum(v)/N] for (k,v) in transf_metrics.items()}

# print(orig_metrics_mean, transf_metrics_mean)

# Python paired sample t-test
from scipy.stats import ttest_rel
def paired_t (a, b):
    np_a = np.array(a)
    np_b = np.array(b)
    s, p = ttest_rel(np.absolute(np_a), np.absolute(np_b))
    return p

def acc_diff (a, b):
    np_a = np.array(a)
    np_b = np.array(b)
    delta = np_a - np_b
    m = statistics.mean(delta)
    s = statistics.stdev(delta)
    return [m, s]

#plot_algo(orig_metrics_mean, transf_metrics_mean,
#           orig_error_metrics, transf_error_metrics, BASELINE)
stat =  {k: [paired_t(transf_metrics[k], v)] for (k,v) in orig_metrics.items()}
# print(stat)

# plt.show()

print("Dataset\t\tFPR_c\tFNR_c\tFPR_a\tFNR_a\n")
print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format("COMPAS original",
        orig_metrics_mean['priv_fpr'][0]*100,
        orig_metrics_mean['priv_fnr'][0]*100,
        orig_metrics_mean['unpriv_fpr'][0]*100,
        orig_metrics_mean['unpriv_fnr'][0]*100
))

print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format("COMPAS de-biased",
        transf_metrics_mean['priv_fpr'][0]*100,
        transf_metrics_mean['priv_fnr'][0]*100,
        transf_metrics_mean['unpriv_fpr'][0]*100,
        transf_metrics_mean['unpriv_fnr'][0]*100
))
