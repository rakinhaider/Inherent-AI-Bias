import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--data',
                      default='compas')
    args.add_argument('-m', '--model_type',
                      default='nb')
    args = args.parse_args()
    dataset = args.data
    model_type = args.model_type
    title = '_'.join([dataset, model_type])
    title = 'outputs/' + title
    data = pd.read_csv(title + '.csv', sep='\t', index_col=0)
    print(data.head())
    # data = data.values

    k_fold = 20

    indices = np.array([3*i for i in range(k_fold)])
    x = range(k_fold)
    plt.plot(x, data.loc[indices][di_col], label='priv')
    plt.plot(x, data.loc[indices + 1][di_col], label='unpriv')
    plt.plot(x, data.loc[indices + 2][di_col], label='overall')
    plt.legend()
    plt.savefig(title + '.pdf', format='pdf')
    plt.show()
