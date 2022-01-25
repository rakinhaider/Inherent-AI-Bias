import argparse

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import \
    load_preproc_data_compas
from aif360.datasets import GermanDataset, BankDataset

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from aif360.sklearn.inprocessing import ExponentiatedGradientReduction
from prejudice_remover import PrejudiceRemover


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma-1', default=2, type=int)
    parser.add_argument('--sigma-2', default=5, type=int)
    parser.add_argument('--mu_p_plus', default=13, type=int)
    parser.add_argument('--mu_u_plus', default=10, type=int)
    parser.add_argument('--delta', default=10, type=int)
    parser.add_argument('--beta', default=1, type=int)
    parser.add_argument('--n-samples', default=10000, type=int)
    parser.add_argument('--n-redline', default=1, type=int)
    parser.add_argument('--n-feature', default=0, type=int)
    parser.add_argument('--tr-rs', default=47, type=int,
                        help='Train Random Seed')
    parser.add_argument('--te-rs', default=41, type=int,
                        help='Test Random Seed')
    parser.add_argument('--estimator', default='nb',
                        choices=['nb', 'lr', 'svm', 'dt', 'pr'],
                        help='Type of estimator')
    parser.add_argument('--reduce', default=False, action='store_true')
    return parser


def get_estimator(estimator, reduce):
    if reduce:
        return ExponentiatedGradientReduction

    if estimator == 'nb':
        return GaussianNB
    elif estimator == 'lr':
        return LogisticRegression
    elif estimator == 'svm':
        return SVC
    elif estimator == 'dt':
        return DecisionTreeClassifier
    elif estimator == 'nn':
        pass
    elif estimator == 'pr':
        return PrejudiceRemover


def get_dataset(dataset_name):
    if dataset_name == 'compas':
        dataset = load_preproc_data_compas(protected_attributes=['race'])
    elif dataset_name == 'german':
        dataset = GermanDataset(
            # this dataset also contains protected
            # attribute for "sex" which we do not
            # consider in this evaluation
            protected_attribute_names=['age'],
            # age >=25 is considered privileged
            privileged_classes=[lambda x: x >= 25],
            # ignore sex-related attributes
            features_to_drop=['personal_status', 'sex'],
        )
    elif dataset_name == 'bank':
        dataset = BankDataset(
            # this dataset also contains protected
            protected_attribute_names=['age'],
            privileged_classes=[lambda x: x >= 25],  # age >=25 is considered privileged
        )
        dataset.metadata['protected_attribute_maps'] = [{1.0: 'yes', 0.0: 'no'}]
        temp = dataset.favorable_label
        dataset.favorable_label = dataset.unfavorable_label
        dataset.unfavorable_label = temp
    else:
        raise ValueError('Dataset name must be one of '
                         'compas, german, bank')
    return dataset