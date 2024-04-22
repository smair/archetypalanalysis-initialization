import os.path

import numpy as np
import matplotlib.pylab as plt

from plot_variables import M, STYLE, STYLE_M, n_for, K_for, repetitions_for, LABEL_RENAME, DATASET_RENAME

################################################################################
# time plot
# This is Figure 6 of the paper.

datasets = ['california','covertype']
INIT = ['Uniform','FurthestFirst','FurthestSum','AAcoreset','KMpp','AApp','AAppMC']
m_ = [20,10,5,1]
for j, m in enumerate(m_):
    STYLE[f'AAppMC{m}'] = STYLE['AAppMC'].copy()
    STYLE[f'AAppMC{m}']['ls'] = STYLE_M[j]
STYLE['AAppMC1']['color']  = 'cyan'

INIT_ = INIT[:-1]+list(map(lambda x: x[0]+str(x[1]), list(zip(('AAppMC',)*len(m_), m_))))

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 12})

show_seeds = False
preprocessing = 'CenterAndMaxScale'
#preprocessing = 'Standardize'
max_iterations = 30
fig, axv = plt.subplots(1, 2, figsize=(8,4))
for ax, dataset in zip(axv, datasets):
    dom = K_for[dataset]
    ax.set_title(f'{DATASET_RENAME[dataset]}')
    ax.set_xlabel(r'$k$')
    if dataset == 'california':
        ax.set_ylabel('Initialization Time in Seconds')
    ax.set_yscale('log')
    ax.set_xticks(dom)
    ax.set_xticklabels(dom)

    for init in INIT_:
        init_time = np.zeros((repetitions_for[dataset], len(dom)))
        AA_time = np.zeros((repetitions_for[dataset], len(dom)))
        for j, k in enumerate(dom):
            m = 0
            filename = f'result_{dataset}_{k}_{init}_{m}_{preprocessing}_{repetitions_for[dataset]}_{max_iterations}_{M}.npz'

            if 'MC' in init:
                tmp = init.split('MC')
                init_ = tmp[0]+'MC'
                m_percent = int(tmp[1])
                # convert a percentage to a chain length
                m = int(np.ceil(n_for[dataset]/100*m_percent))
                filename = f'result_{dataset}_{k}_{init_}_{m}_{preprocessing}_{repetitions_for[dataset]}_{max_iterations}_{M}.npz'

            location = os.path.join('results', filename)
            if not os.path.exists(location):
                print(f'file not found: {location}')
                continue
            res_data = np.load(location)
            res_time_init = res_data['res_time_init']
            res_rss_init = res_data['res_rss_init'] / n_for[dataset]
            res_time_AA = res_data['res_time_AA']
            res_rss_AA = res_data['res_rss_AA'] / n_for[dataset]

            if res_data['res_rss_AA'].shape[0] != repetitions_for[dataset]:
                print(f'run not complete: {location}')
                continue

            init_time[:,j] = res_time_init
            AA_time[:,j] = res_time_AA[:,-1] # take the last one and divide by number of iterations

            median_time_per_iteration = np.median(np.diff(res_time_AA))
            print(f'{filename} init={np.median(res_time_init)} per-it={median_time_per_iteration}')

        init_time_median = np.median(init_time, axis=0)
        init_time_quantile_upper = np.quantile(init_time, q=0.75, axis=0)
        init_time_quantile_lower = np.quantile(init_time, q=0.25, axis=0)

        ax.fill_between(dom, init_time_quantile_upper, init_time_quantile_lower,
                        alpha=0.2, **STYLE[init])
        if show_seeds:
            for i in range(repetitions_for[dataset]):
                ax.plot(dom, init_time[i], alpha=0.2, **STYLE[init])
        ax.plot(dom, init_time_median, label=LABEL_RENAME[init], **STYLE[init])

        if init=='AApp':
            ax.plot(dom, np.median(AA_time, axis=0)/max_iterations, 'k--', label=r'$1\times$ AA Iter.')
            ax.plot(dom, np.multiply(np.median(AA_time, axis=0)/max_iterations, dom), 'k-', label=r'$k\times$ AA Iter.')

        ax.grid(True, which='both')

        if dataset == 'california':
            ax.legend(bbox_to_anchor=(0, 1.165, 2.1775, 0.2), loc="lower left",
                      mode="expand", borderaxespad=0, ncol=4)

fig.tight_layout()


fig.savefig(f'/tmp/time_{preprocessing}.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig(f'/tmp/time_{preprocessing}.png', dpi=300, transparent=False, bbox_inches='tight')
