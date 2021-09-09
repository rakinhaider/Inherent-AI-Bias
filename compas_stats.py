from aif360.datasets.compas_dataset import default_preprocessing
import pandas as pd
from aif360.algorithms.preprocessing.optim_preproc_helpers.\
    data_preproc_functions import load_preproc_data_compas


def get_base_rates(dataset):
    df, _ = dataset.convert_to_dataframe()
    print('Base rate in entire dataset.')
    print(df['two_year_recid'].value_counts()/len(df))
    grouped = df.groupby(dataset.protected_attribute_names)
    for r, grp in grouped:
        print('Race: ', r)
        print('Favored and unfavored counts :\n', grp[dataset.label_names].value_counts())
        print('Percentage of favored and unfavored :\n', grp[dataset.label_names].value_counts()/len(grp))
        print('Total :', len(grp))


if __name__ == "__main__":
    # Load the raw file.
    df = pd.read_csv('C:/Python38_64/Lib/site-packages/aif360/data/raw/compas/compas-scores-two-years.csv')

    dataset = load_preproc_data_compas(protected_attributes=['race'])
    get_base_rates(dataset)

