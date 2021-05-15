import argparse
import pandas as pd
import constants as c
import numpy as np


def get_di(clf, alpha=0.5):
    num = clf[c.u_tp_col] * alpha + clf[c.u_fp_col] *(1 - alpha)
    denom = clf[c.p_tp_col] * alpha + clf[c.p_fp_col] *(1 - alpha)
    return num/denom


def get_modified_di(clf_a, clf, alpha=0.5):
    num = clf_a[c.u_tp_col] * alpha + clf_a[c.u_fp_col] * (1 - alpha)
    delta_u_pos = clf_a[c.u_tp_col] - clf[c.u_tp_col]
    delta_u_neg = clf_a[c.u_fp_col] - clf[c.u_fp_col]
    denom = clf_a[c.p_tp_col] * alpha + clf_a[c.p_fp_col] * (1 - alpha)
    delta_p_pos = clf_a[c.p_tp_col] - clf[c.p_tp_col]
    delta_p_neg = clf_a[c.p_fp_col] - clf[c.p_fp_col]

    print(num, delta_u_pos, delta_u_neg)
    print(denom, delta_p_pos, delta_p_neg)

    print(num - alpha * delta_u_pos - (1 - alpha) * delta_u_neg)
    print(denom - alpha * delta_p_pos - (1 - alpha) * delta_p_neg)
    print()
    print(delta_u_neg + delta_u_pos)
    print(delta_p_neg + delta_p_pos)
    print((delta_u_neg + delta_u_pos)/(delta_p_neg + delta_p_pos))
    print(num/denom)

    assert (delta_u_neg + delta_u_pos)/(delta_p_neg + delta_p_pos) > num/denom
    num = num - alpha * delta_u_pos - (1 - alpha) * delta_u_neg
    denom = denom - alpha * delta_p_pos - (1 - alpha) * delta_p_neg
    return num / denom


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--data',
                      default='compas')
    args.add_argument('-m', '--model_type',
                      default='nb')
    args = args.parse_args()
    title = '_'.join([args.data, args.model_type])
    title = 'outputs/' + title
    data = pd.read_csv(title + '.csv', sep='\t', index_col=0)

    k_fold = 20

    # indices of theta_p classifiers.
    indices = np.array([3 * i for i in range(k_fold)])
    columns = [c.p_tp_col, c.p_fp_col, c.u_tp_col, c.u_fp_col]
    print(data[columns].head(10))

    u_tp_theta_p = data.loc[indices][c.u_tp_col].values
    u_tp_theta = data.loc[indices+2][c.u_tp_col].values
    delta_u_pos = u_tp_theta_p - u_tp_theta

    u_fp_theta_p = data.loc[indices][c.u_fp_col].values
    u_fp_theta = data.loc[indices + 2][c.u_fp_col].values
    delta_u_neg = u_fp_theta_p - u_fp_theta

    p_tp_theta_p = data.loc[indices][c.p_tp_col].values
    p_tp_theta = data.loc[indices + 2][c.p_tp_col].values
    delta_p_pos = p_tp_theta_p - p_tp_theta

    p_fp_theta_p = data.loc[indices][c.p_fp_col].values
    p_fp_theta = data.loc[indices + 2][c.p_fp_col].values
    delta_p_neg = p_fp_theta_p - p_fp_theta

    # print(delta_p_pos + delta_p_neg)
    # print(delta_u_pos + delta_u_neg)

    # print(delta_p_pos + delta_p_neg < delta_u_pos + delta_u_neg)

    for k in range(k_fold):
        theta_p = data.loc[3*k]
        theta_u = data.loc[3*k + 1]
        theta = data.loc[3*k + 2]
        #  print(theta_p)
        # print(theta)
        di_theta_p = get_di(theta_p)
        di_theta = get_di(theta)
        di_theta_u = get_di(theta_u)
        print(di_theta, di_theta_p, di_theta_u, sep='\t\t')
        if di_theta_p > di_theta and di_theta_p <= 1:
            print(get_modified_di(theta_p, theta))

        if di_theta_u > di_theta and di_theta_u <= 1 :
            print(get_modified_di(theta_u, theta))
