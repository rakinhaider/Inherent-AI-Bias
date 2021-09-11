import matplotlib.pyplot as plt
import pandas as pd

def plot_algo(orig_metrics_mean, transf_metrics_mean, 
              orig_error_metrics, transf_error_metrics, 
              model_type):
    pd.set_option('display.multi_sparse', False)
    plt.rcParams.update({'font.size': 8}) # must set in top

    results = [orig_metrics_mean,
            transf_metrics_mean]


    errors = [orig_error_metrics,
            transf_error_metrics]

    index = pd.Series([model_type+'_orig']+ [model_type+'_syn'], name='Classifier Bias Mitigator')

    df = pd.concat([pd.DataFrame(metrics) for metrics in results], axis=0).set_index(index)
    df_error = pd.concat([pd.DataFrame(metrics) for metrics in errors], axis=0).set_index(index)
    ax = df.plot.bar(yerr=df_error, capsize=4, rot=0, subplots=True, title=['','','','','', '', '', '', '', ''], fontsize = 12)
    plot1 = ax[0]
    plot1.set_ylim=([0, 0.8])
    plot2 = ax[1]
    plot2.set_ylim=([-0.5, 0])
    plot3 = ax[2]
    plot3.set_ylim=([0, 1])
    plot4 = ax[3]
    plot4.set_ylim=([-0.5, 0])
    plot5 = ax[4]
    plot5.set_ylim=([-0.5, 0])
    plot5 = ax[5]
    plot5.set_ylim=([0, 0.2])

    plt.savefig('./eps/'+model_type+'_orig_synth.eps', format='eps')
    # print(df)

