import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd


def get_ellipse_prop(dist, delta, priv, label):
    if label:
        if priv:
            center = (dist['mu_ps']['p'],
                      dist['mu_ps']['u'] - delta//2)
        else:
            center = (dist['mu_ps']['p'] - delta // 2,
                      dist['mu_ps']['u'])
    else:
        if priv:
            center = (dist['mu_ns']['p'],
                      dist['mu_ns']['u'] + delta//2)
        else:
            center = (dist['mu_ns']['p'] + delta // 2,
                      dist['mu_ns']['u'])

    width = dist['sigma_ps']['p']
    height = dist['sigma_ps']['u']
    chisqr_len = np.sqrt(5.991)
    return center, 2 * width * chisqr_len, 2 * height * chisqr_len


def plot_decision_boundaries(X, y, s, model,
                             plot_by_group=False,
                             is_trained=False,
                             dist=None,
                             delta=None,
                             **model_params):
    """
    Function to plot the decision boundaries of a classification model.
    This uses just the first two columns of the data for fitting
    the model as we need to find the predicted value for every point in
    scatter plot.
    Arguments:
            X: Feature data as a NumPy-type array.
            y: Label data as a NumPy-type array.
            model: A Scikit-learn ML estimator class
            e.g. GaussianNB (imported from sklearn.naive_bayes) or
            LogisticRegression (imported from sklearn.linear_model)
            **model_params: Model parameters to be passed on to the ML estimator

    Typical code example:
            plt.figure()
            plt.title("KNN decision boundary with neighbros: 5",fontsize=16)
            plot_decision_boundaries(X_train,y_train,KNeighborsClassifier,n_neighbors=5)
            plt.show()
    """
    try:
        X = np.array(X)
        y = np.array(y).flatten()
        s = np.array(s).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")
    # Reduces to the first two columns of data
    reduced_data = X[:, :2]
    if not is_trained:
        # Instantiate the model object
        model = model(**model_params)
        # Fits the model with the reduced data
        model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    if plot_by_group:
        df = pd.DataFrame(np.concatenate((reduced_data, s.reshape(-1, 1),
                                          y.reshape(-1, 1)), axis=1))
        grouped = df.groupby(by=[2, 3])
        mp = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        cdist = {(0, 0): 'red', (0, 1): 'green',
                 (1, 0): 'lightcoral', (1, 1): 'lime'}
        grouped = sorted(grouped, key=lambda tup: tup[0][1], reverse=False)
        for (gs, gy), grp in grouped:
            plt.scatter(grp[0], grp[1], c=[cdist[(gs, gy)] for i in range(len(grp))],
                        label='{}, {}'.format('p' if gs else 'u',
                                              '+' if gy else '-'),
                        alpha=0.8)

            center, width, height = get_ellipse_prop(dist, delta, gs, gy)
            ellipse = Ellipse(xy=center, width=width, height=height,
                              color='yellow' if gy == 1 else 'darkred',
                              fill=False, linewidth=2)
            plt.gca().add_patch(ellipse)
        plt.legend()
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.xlabel("Feature-1", fontsize=15)
    plt.ylabel("Feature-2", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return plt