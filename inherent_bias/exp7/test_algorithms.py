from mitigators import BaseMitigator, NullMitigator, SyntheticMitigator

class TestAlgorithms:

    def __init__(self, model_type):
        self.model_type = model_type

    def run_original(self, dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, orig_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER):
        # print('\n------------------------------\n')
        # print('[INFO] Original Results......')
        # print('\n------------------------------\n')

        null_mitigator = NullMitigator()
        orig_metrics = null_mitigator.run_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, orig_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER)

        #null_mitigator.run_explainer(dataset_orig_train, dataset_orig_test, model_type, SCALER)
        return orig_metrics


    def run_oversample(self,dataset_orig_train, dataset_orig_val, dataset_orig_test, privileged_groups, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, model_type, transf_metrics, f_label, uf_label, THRESH_ARR, DISPLAY, OS_MODE, SCALER):
        # print('\n------------------------------\n')
        # print('[INFO] Random Oversampling ......')
        # print('\n------------------------------\n')
        synth_mitigator = SyntheticMitigator()
        metric_transf_train, transf_metrics = synth_mitigator.run_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, privileged_groups, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, model_type, transf_metrics, f_label, uf_label, THRESH_ARR, DISPLAY, OS_MODE, SCALER)

        #synth_mitigator.run_explainer(dataset_orig_train, dataset_orig_val, dataset_orig_test, privileged_groups, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, model_type, transf_metrics, f_label, uf_label, THRESH_ARR, DISPLAY, OS_MODE, SCALER)
        return metric_transf_train, transf_metrics
