import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.special import erf
from inherent_bias.utils import get_c123
import argparse


def set_rcparams(**kwargs):
    fontsize = kwargs.get('fontsize', 10)
    params = {
        'text.usetex': True,
        'pdf.fonttype': 42,
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": 'medium',
        "xtick.labelsize": 'x-small',
        "ytick.labelsize": 'x-small',
        "mathtext.fontset": 'cm',
        "mathtext.default": 'bf',
        # "figure.figsize": set_size(width, fraction)
    }
    if kwargs.get('titlepad') is not None:
        params["axes.titlepad"] = kwargs.get('titlepad')
    if kwargs.get('labelpad') is not None:
        params["axes.labelpad"] = kwargs.get('labelpad')
    if kwargs.get('markersize') is not None:
        params["lines.markersize"] = kwargs.get('markersize')
    matplotlib.rcParams.update(params)


# Source: https://tobiasraabe.github.io/blog/matplotlib-for-publications.html
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


def plot_normal(mu, sigma, ax, label=None):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), label=label)


def plot_feat_dist(dist, **kwargs):
    width = kwargs.get('width', 462)
    fraction = kwargs.get('fraction', 1)
    aspect_ratio = kwargs.get('aspect_ratio', 'golden')
    f, ax = plt.subplots(2, 2,
        figsize=set_size(width, fraction, aspect_ratio=aspect_ratio)
    )
    shift_p = 0
    shift_u = 0
    if kwargs.get('shift_priv', False) == True:
        shift_u = kwargs['shift_random']
    else:
        shift_p = kwargs.get('shift_random', 0)
    avg_p = (dist['mu_ps']['p'] + dist['mu_ns']['p']) / 2 + shift_p
    avg_u = (dist['mu_ps']['u'] + dist['mu_ns']['u']) / 2 + shift_u
    sigma_p = dist['sigma_ps']['p']
    sigma_u = dist['sigma_ps']['u']
    p_range = (min(dist['mu_ps']['p'], dist['mu_ns']['p']) - 3 * sigma_p,
               max(dist['mu_ps']['p'], dist['mu_ns']['p']) + 3 * sigma_p
               )
    u_range = (min(dist['mu_ps']['u'], dist['mu_ns']['u']) - 3 * sigma_u,
               max(dist['mu_ps']['u'], dist['mu_ns']['u']) + 3 * sigma_u
               )

    combs = [
        [
            [(dist['mu_ps']['p'], dist['mu_ns']['p']), (sigma_p, sigma_p),
             p_range],
            [(avg_u, avg_u), (sigma_u, sigma_u), u_range]
        ],
        [
            [(avg_p, avg_p), (sigma_p, sigma_p), p_range],
            [(dist['mu_ps']['u'], dist['mu_ns']['u']), (sigma_u, sigma_u),
             u_range]
        ]
    ]

    for i, row in enumerate(ax):
        for j, a in enumerate(row):
            mus = combs[i][j][0]
            sigmas = combs[i][j][1]
            plot_normal(mus[0], sigmas[0], a)
            plot_normal(mus[1], sigmas[1], a)
            left, right = combs[i][j][2]
            a.set_xlim(left, right)
            a.set_ylim(-0.01, 0.22)

    ax[0][0].set_title('Test Scores')
    ax[0][1].set_title('GPA')

    ax[0][0].set_ylabel('Privileged')
    ax[1][0].set_ylabel('Unprivileged')

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.25)


def plot_erf(sigma_1, sigma_2, delta, alpha):
    erfs = []
    start = - 250
    end = 250
    for i in range(start, end + 1):
        erfs.append(erf(i / 100))

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.plot([i / 100 for i in range(start, end + 1)], erfs, color='black')

    calpha = np.log(alpha / (1 - alpha))
    c1, c2, c3 = get_c123(sigma_1, sigma_2, delta)
    points = [c1, c3, c1 + c2 * calpha, c3 + c2 * calpha, c3 - c2 * calpha,
              c1 - c2 * calpha]
    plt.plot(points, [erf(p) for p in points], 'o', color='black')

    texts = [r'$c_1$', r'$c_3$', r'$c_1+c_2*c_\alpha$', r'$c_3+c_2*c_\alpha$',
             r'$c_3 - c_2*c_\alpha$', r'$c_1 - c_2*c_\alpha$']

    ax = plt.gca()

    ax.annotate(texts[0], (points[0] - 0.2, erf(points[0]) + 0.05),
                rotation=-75)
    ax.annotate(texts[1], (points[1] - 0.15, erf(points[1]) + 0.1),
                rotation=-75)
    ax.annotate(texts[2], (points[2] + 0.02, erf(points[2]) - 1.3),
                rotation=-75)
    ax.annotate(texts[3], (points[3] - 0.4, erf(points[3]) + 0.07),
                rotation=-75)

    ax.annotate(texts[4], (points[4] - 0.05, erf(points[4]) - 1.3),
                rotation=-75)
    ax.annotate(texts[5], (points[5], erf(points[5]) - 1.3),
                rotation=-75)

    plt.ylim(-1.1, 1.3)
    plt.xlim(-2.5, 2.5)
    plt.xlabel('x')
    plt.ylabel('erf(x)')
    plt.tight_layout()

    mid_points = [(c1 + c3) / 2, (c1 + c3) / 2 + c2 * calpha,
                  (c1 + c3) / 2 - c2 * calpha]
    plt.plot(mid_points, [erf(i) for i in mid_points], 'o', color='black')

    plt.plot([c1 + c2 * calpha, c3 + c2 * calpha],
             [erf(i) for i in [c1 + c2 * calpha, c3 + c2 * calpha]],
             color='black')
    plt.plot([c1 + c2 * calpha, c1 + c2 * calpha],
             [erf(c1 + c2 * calpha), erf(c3 + c2 * calpha)], color='black')
    plt.plot([c3 + c2 * calpha, c1 + c2 * calpha],
             [erf(c3 + c2 * calpha), erf(c3 + c2 * calpha)], color='black')

    plt.plot([c1 - c2 * calpha, c3 - c2 * calpha],
             [erf(c1 - c2 * calpha), erf(c3 - c2 * calpha)], color='black')
    plt.plot([c1 - c2 * calpha, c1 - c2 * calpha],
             [erf(c1 - c2 * calpha), erf(c3 - c2 * calpha)], color='black')
    plt.plot([c3 - c2 * calpha, c1 - c2 * calpha],
             [erf(c3 - c2 * calpha), erf(c3 - c2 * calpha)], color='black')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--what', choices=['erf', 'featdist'], default='erf')
    parser.add_argument('--sigma-1', default=2, type=int)
    parser.add_argument('--sigma-2', default=5, type=int)
    parser.add_argument('--mu_p_plus', default=13, type=int)
    parser.add_argument('--mu_u_plus', default=10, type=int)
    parser.add_argument('--delta', default=10, type=int)
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--width', default=239, type=int)
    args = parser.parse_args()
    # textwidth = 505
    linewidth = args.width
    what = args.what
    out_dir = 'outputs/figures/'

    if what == 'erf':
        set_rcparams(fontsize=10, titlepad=3,
                     labelpad=1, markersize=4)

        plt.gcf().set_size_inches(set_size(linewidth, 0.95, 0.6))
        plot_erf(args.sigma_1, args.sigma_2, args.delta, args.alpha)
        fname = 'erf_plot.pdf'

    elif what == 'featdist':
        set_rcparams(fontsize=7)
        mu_p_plus, mu_u_plus, = args.mu_p_plus, args.mu_u_plus
        mu_p_minus, mu_u_minus = mu_p_plus - args.delta, mu_u_plus - args.delta
        sigma_p, sigma_u = args.sigma_1, args.sigma_2
        dist = {
            'mu_ps': {'p': mu_p_plus, 'u': mu_u_plus},
            'sigma_ps': {'p': sigma_p, 'u': sigma_u},
            'mu_ns': {'p': mu_p_minus, 'u': mu_u_minus},
            'sigma_ns': {'p': sigma_p, 'u': sigma_u}
        }
        plot_feat_dist(dist, width=linewidth,
                       fraction=0.95, aspect_ratio=0.7)
        fname = '{}-{}-{}-{}-{}-{}.pdf'.format(
            dist['mu_ps']['p'], dist['mu_ps']['u'],
            dist['mu_ns']['p'], dist['mu_ns']['u'],
            dist['sigma_ps']['p'], dist['sigma_ps']['u'])

    plt.savefig(out_dir + fname, format='pdf')