from aif360.metrics import (
                            BinaryLabelDatasetMetric,
                            ClassificationMetric
)
from matplotlib.colors import to_hex
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np


def get_dataset_metrics(fd_train,
                         unprivileged_groups,
                         privileged_groups,
                         verbose=False):

    metrics = BinaryLabelDatasetMetric(fd_train,
                   unprivileged_groups=unprivileged_groups,
                   privileged_groups=privileged_groups)
    
    if verbose:
        print('Mean Difference:', metrics.mean_difference())
        print('Dataset Base Rate', metrics.base_rate(privileged=None))
        print('Privileged Base Rate', metrics.base_rate(privileged=True))
        print('Protected Base Rate', metrics.base_rate(privileged=False))
        print('Disparate Impact:', metrics.disparate_impact())
        # print('Confusion Matrix:', metrics.binary_confusion_matrix())
        
    return metrics.mean_difference(), metrics.disparate_impact()


def get_classifier_metrics(clf, data,
                      unprivileged_groups,
                      privileged_groups,
                      verbose=False):

    data_pred = data.copy()
    data_x, data_y = data.get_xy(keep_protected=False)
    data_pred.labels = clf.predict(data_x)
    
    metrics = ClassificationMetric(data, 
                                   data_pred, 
                                   privileged_groups=privileged_groups, 
                                   unprivileged_groups=unprivileged_groups)
    
    if verbose:
        print('Mean Difference:', abs(0-metrics.mean_difference()))
        print('Disparate Impact:', abs(1 - metrics.disparate_impact()))
        # print('Confusion Matrix:', metrics.binary_confusion_matrix())
        print('Accuracy:', metrics.accuracy())

        
    return [metrics.mean_difference(), 
            metrics.disparate_impact(), 
            metrics.accuracy(),
            metrics.accuracy(privileged=True),
            metrics.accuracy(privileged=False)]

def plot_lr_boundary(clf, plt, alpha, beta, di):
    # Retrieve the model parameters.
    b = clf.intercept_[0]
    w1, w2 = clf.coef_.T
    # Calculate the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2

    # Plot the data and the classification with the decision boundary.
    xmin, xmax = -10, 20
    ymin, ymax = -10, 25
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    label = str(alpha) + '_' + str(beta) + '_' + str(di) 
    plt.plot(xd, yd, lw=1, label=label)
    
def get_model_properties(model):
    if isinstance(model, DecisionTreeClassifier):
        return model.get_depth()
    elif isinstance(model, LogisticRegression):
        return model.coef_
    elif isinstance(model, GaussianNB):
        return model.theta_, np.sqrt(model.sigma_)