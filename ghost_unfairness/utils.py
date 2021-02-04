from aif360.metrics import (
                            BinaryLabelDatasetMetric,
                            ClassificationMetric
)


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

        
    return metrics.mean_difference(), metrics.disparate_impact(), metrics.accuracy()