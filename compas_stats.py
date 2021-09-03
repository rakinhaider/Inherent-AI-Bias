from aif360.datasets.compas_dataset import default_preprocessing
import pandas as pd
from aif360.algorithms.preprocessing.optim_preproc_helpers.\
    data_preproc_functions import load_preproc_data_compas

from mixed_model_experiment import get_base_rates


def compas_stats(df):
    label = 'two_year_recid'
    # Preprocess and drops samples not used in propublica analysis.
    # aif360 always applies this default processing.
    df = default_preprocessing(df)
    print(df['race'].value_counts())
    grouped = df.groupby(by=['race'])
    for r, grp in grouped:
        print()
        print('Number of Person in favored(0) and unfavored(1) class in', r)
        print(grp[label].value_counts())
        print('Percentage of samples in favored(0) and unfavored(1) class.')
        print(grp[label].value_counts()/len(grp))
    return df


if __name__ == "__main__":
    # Load the raw file.
    df = pd.read_csv('C:/Python38_64/Lib/site-packages/aif360/data/raw/compas/compas-scores-two-years.csv')

    # compas_stats(df)

    dataset = load_preproc_data_compas(protected_attributes=['race'])
    get_base_rates(dataset)

