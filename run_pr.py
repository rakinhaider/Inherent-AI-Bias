### Dummy test Not actual
import numpy as np
from aif360.sklearn.datasets.compas_dataset import fetch_compas
from prejudice_remover import PrejudiceRemover
from predictive_power_difference import encode_categories

if __name__ == "__main__":
    X, y = fetch_compas(binary_race=True,
                        dropcols=['sex', 'age', 'c_charge_desc'])

    X, y = encode_categories('compas', X, y)
    print(y.head())
    y.name = 'two_years_recid'
    pr = PrejudiceRemover(sensitive_attr='race', class_attr='two_years_recid',
                          favorable_label=0,
                          all_sensitive_attributes=['sex', 'race'],
                          privileged_value=0.0)

    pr.fit(X.head(5), y.head(5))
    print(pr.predict(X.head(5)))
    print(pr.predict_proba(X.head(5)))