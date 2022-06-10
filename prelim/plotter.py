import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def set_size(width, fraction=1, aspect_ratio='golden'):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    if aspect_ratio == 'golden':
        aspect_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * aspect_ratio

    return fig_width_in, fig_height_in


def plot_bar_graphs(data, ranges, colors, alphas, measures,):
    x = [1, 2]
    fig, ax = plt.subplots(3, 3, figsize=plt.gcf().get_size_inches())

    legends = get_legends(measures, colors)
    for i, alpha in enumerate(alphas):
        ax[i][0].set_ylabel(r'$\alpha =$ ' + '{:.2f}'.format(alpha))
        for j, measure in enumerate(measures):
            measure = measure
            cur_ax = ax[i][j]
            cur_ax.bar(x, data.iloc[i][
                [measure.lower() + 'p', measure.lower() + 'u']],
                       color=colors[j], edgecolor='black', width=1)
            cur_ax.set_ylim(ranges[j][i][0], ranges[j][i][1])
            cur_ax.set_xticks([1.5])
            cur_ax.set_xticklabels(['NBC'])
            annotate_bars(cur_ax, afont_size=10, gap=0)

    plt.tight_layout()
    fig.legend(handles=legends, loc='center right')
    plt.subplots_adjust(right=0.85)


def annotate_bars(cur_ax, afont_size=10, arotation=0, gap=1.5):
    for bar in cur_ax.patches:
        cur_ax.annotate(format(bar.get_height(), '.1f'),
                        (bar.get_x() + bar.get_width() / 2,
                         bar.get_height()+gap), ha='center', va='center',
                        size=afont_size, xytext=(0, 8),
                        textcoords='offset points',
                        rotation=arotation)


def get_legends(measures, colors):
    legends = []
    for i, m in enumerate(measures):
        for j, g in enumerate(['p', 'u']):
            p = mpatches.Patch(color=colors[i][j],
                               label=r'${}_{}$'.format(m, g))
            legends.append(p)
    return legends


def plot_grouped_bars(data, ranges, colors, measures,
                      x_key='alpha', y_key='method',
                      selection_criteria={},
                      **kwargs):

    xs = kwargs[x_key+'s']
    data = data.copy(deep=True)
    for key in selection_criteria:
        data = data[data[key].isin(selection_criteria[key])]
    columns = [data[x_key].values, data[y_key].values]
    columns = pd.MultiIndex.from_tuples(list(zip(*columns)))
    data.drop(columns=[y_key], inplace=True)
    data = pd.DataFrame(data.T.values, columns=columns, index=data.columns)
    print(data)

    legends = get_legends(measures, colors)

    fig, ax = plt.subplots(len(xs), len(measures),
                           figsize=plt.gcf().get_size_inches())
    if ax.ndim == 1:
        ax = [ax]
    for i, val in enumerate(xs):
        if x_key == 'alpha':
            ax[i][0].set_ylabel(r'$\alpha =$ ' + '{:.2f}'.format(val))
        else:
            ax[i][0].set_ylabel(x_key + ' = {}'.format(val))

    for j, measure in enumerate(measures):
        cur_data = data.loc[[measure.lower() + 'p', measure.lower() + 'u']]
        for i, cols in enumerate(xs):
            cur_ax = ax[i][j]
            cur_data[cols].T.plot(ax=cur_ax, kind='bar',
                                  color=colors[j],
                                  width=.8, edgecolor='black')
            cur_ax.legend_.remove()
            cur_ax.set_ylim(ranges[j][i][0], ranges[j][i][1])
            cur_ax.set_axisbelow(True)
            cur_ax.set_xlabel(y_key)
            if kwargs.get('force_gap', False):
                gap = kwargs['gap']
            else:
                gap = (ranges[j][i][1] - ranges[j][i][0]) * 0.05

            annotate_bars(cur_ax, afont_size=kwargs['afont_size'],
                          arotation=kwargs['arotation'],
                          # gap=kwargs['gap'])
                          gap=gap)
            for tick in cur_ax.get_xticklabels():
                tick.set_rotation(0)
    plt.tight_layout()
    fig.legend(handles=legends, loc='center right')
    plt.subplots_adjust(right=0.85)


if __name__ == "__main__":

    what = 'compas_mode' # choices 'eq', 'grp', 'res', 'compas_mode'
    fname = ""
    if what == 'res':
        fname = 'res_ls_n_2'
        headers = ['res', 'alpha', 'accp', 'accu',
                   'srp', 'sru', 'fprp', 'fpru']
    elif what == 'grp':

        fname = 'ls_n_2'
        headers = ['method', 'alpha', 'accp', 'accu',
                   'srp', 'sru', 'fprp', 'fpru']
    elif what == 'eq':
        fname = 'eq_n_2'
        headers = ['model', 'alpha', 'accp', 'accu',
                   'srp', 'sru', 'fprp', 'fpru']
    elif what == 'compas_mode':
        fname = 'compas_mode'
        headers = ['mode', 'alpha', 'accp', 'accu',
                   'srp', 'sru', 'fprp', 'fpru']
    print(fname)
    data = pd.read_csv('{}.txt'.format(fname), sep='\t', header=None)
    data.columns = headers
    print(data)

    plt.gcf().set_size_inches(set_size(280, 280 / 140))

    measures = ['ACC', 'SR', 'FPR']
    alphas = [0.25, 0.5, 0.75]
    afont_size = 10
    ranges = [[[80, 105], [80, 105], [80, 105]],
              [[10, 30], [45, 55], [70, 90]],
              [[0, 20], [0, 40], [0, 75]]]
    colors = [['springgreen', 'lightseagreen'],
              ['lightcoral', 'firebrick'],
              ['cyan', 'royalblue']]
    if what == 'eq':
        ranges = [[[90, 105]],
                  [[20, 90]],
                  [[0, 30]]]
        plot_grouped_bars(data, ranges, colors, measures,
                          x_key='model', y_key='alpha',
                          models=['NBC'],
                          afont_size=10,
                          arotation=90,
                          gap=2.5)
    elif what == 'grp':
        ranges = [[[80, 110], [80, 110], [80, 110]],
                  [[10, 100], [10, 100], [10, 100]],
                  [[0, 75], [0, 75], [0, 75]]]
        afont_size = 7
        plot_grouped_bars(data, ranges, colors, measures,
                          x_key='method', y_key='alpha',
                          methods=['NBC', 'PR', 'RBC'],
                          alphas=alphas, afont_size=afont_size,
                          arotation=0, gap=0, force_gap=True)
    elif what == 'res':
        ranges = [[[30, 110], [30, 110], [30, 110]],
                  [[0, 30], [78, 105], [0, 110]],
                  [[0, 5], [35, 115], [0, 110]]]
        plot_grouped_bars(data, ranges, colors, measures,
                          x_key='res', y_key='alpha',
                          ress=['Low', 'High'],
                          afont_size=10,
                          arotation=90,
                          gap=2.5)

    elif what == 'compas_mode':
        ranges = [[[50, 90], [50, 90], [0, 100]],
                  [[5, 110], [0, 110], [0, 110]],
                  [[0, 110], [0, 110], [0, 110]]]
        plot_grouped_bars(data, ranges, colors, measures,
                          x_key='mode', y_key='alpha',
                          modes=['Priv. Unfavored', 'Unpriv. Favored'],
                          afont_size=8,
                          arotation=90,
                          gap=5)

    dir = 'D:/Thesis/MachineFairness/ContexualFairness/2022-1Spring/Prelim/' \
          'PrelimSlides/figures/feature_disparity/'
    plt.savefig(dir+'{}.pdf'.format(fname), format='pdf')
    plt.show()


