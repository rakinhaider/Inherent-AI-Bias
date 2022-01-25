import argparse
import os
import numpy
import pandas as pd
from aif360.sklearn.datasets import (
    fetch_compas, fetch_bank, fetch_german
)
from pandas import CategoricalDtype
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from inherent_bias.utils import get_datasets
from sklearn.model_selection import cross_val_score


def get_dataset_properties(dataset):
    if dataset == 'fair':
        prot_attr = 'race'
        label = 'label'
        orders = {prot_attr: [1.0, 0.0],
                  label: [1.0, 0.0]}
    elif dataset == 'bank':
        prot_attr = 'age'
        label = 'deposit'
        orders = {prot_attr: ['(0, 35]', '(35, 100]'],
                  label: ['yes', 'no']}
    elif dataset == 'german':
        prot_attr = 'sex'
        label = 'credt-risk'
        orders = {prot_attr: ['male', 'female'], label: ['good', 'bad']}
    elif dataset == 'compas':
        prot_attr, label = 'race', 'two_years_recid'
        orders = {
            prot_attr: ['Caucasian', 'African-American'],
            label: ['Survived', 'Recidivated']
        }
    prop = {
        'orders': orders, 'prot_attr': prot_attr, 'label': label
    }
    return prop


def encode_categories(dataset, X, y):
    prop = get_dataset_properties(dataset)
    orders, prot_attr, label = prop['orders'], prop['prot_attr'], prop['label']
    for c, t in zip(X.columns, X.dtypes):
        if c == prot_attr:
            cats = CategoricalDtype(categories=orders[c], ordered=True)
            X[c] = X[c].astype(cats).cat.codes
        elif t == 'category':
            X[c] = X[c].astype('category').cat.codes
    cats = CategoricalDtype(categories=orders[label], ordered=True)
    y = y.astype(cats).cat.codes
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    return X, y


def get_sklearn_dataset(dataset):
    if dataset == 'fair':
        protected = ["race"]

        kwargs = {'protected_attribute_names': protected,
                  'favorable_classes': [1], 'beta': 1, 'n_dep_feat': 0,
                  'ds': True}
        dist = {
            'mu_ps': {'p': 10, 'u': 13}, 'sigma_ps': {'p': 2, 'u': 5},
            'mu_ns': {'p': 0, 'u': 3}, 'sigma_ns': {'p': 2, 'u': 5}
        }
        kwargs['dist'] = dist.copy()
        kwargs['alpha'] = args.alpha
        n_samples, n_feature, n_redline = 10000, 3, 2
        train_fd, _ = get_datasets(n_samples, n_feature, n_redline,
                kwargs, train_random_state=47, test_random_state=43)
        X, y = train_fd.get_xy(keep_protected=True)
        X.reset_index(drop=True, inplace=True)
        y = pd.Series(y, name='label', index=list(X.index))
        print(X, y)
        return X, y
    elif dataset == 'bank':
        X, y = fetch_bank(dropna=True)
        X['age'] = pd.cut(X['age'], [0, 35, 100]).astype(str)
        return encode_categories(dataset, X, y)
    elif dataset == 'german':
        X, y = fetch_german()
        X, y = encode_categories(dataset, X, y)
        return X, y
    elif dataset == 'compas':
        X, y = fetch_compas(binary_race=True,
                            dropcols=['sex', 'age', 'c_charge_desc'])
        X, y = encode_categories(dataset, X, y)
        return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['bank', 'compas', 'german', 'fair'])
    parser.add_argument('--alpha', type=float, default=0.25)
    args = parser.parse_args()
    X, y = get_sklearn_dataset(args.dataset)
    prop = get_dataset_properties(args.dataset)
    prot_attr = prop['prot_attr']
    prot_attr_vals = prop['orders'][prot_attr]
    ppds = pd.Series(dtype=object)

    for c, t in zip(X.columns, X.dtypes):

        if c == prot_attr:
            continue
        if t == numpy.int8:
            clf = CategoricalNB()
        else:
            clf = GaussianNB()
        print(c, t, clf)

        accs = [0, 0]
        for i, prot_attr_val in enumerate(prot_attr_vals):
            sub_X = X[X[prot_attr] == i][[c]]
            sub_y = y[sub_X.index]
            accs[i] = cross_val_score(clf, sub_X, sub_y, cv=10)

        ppds[c] = abs(accs[0].mean() - accs[1].mean())

    ppds.sort_values(ascending=False, inplace=True)
    ppds.to_csv(os.path.join('outputs/ppds', args.dataset + '.csv'),
                sep='\t')