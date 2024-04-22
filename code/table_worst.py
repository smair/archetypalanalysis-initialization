import os.path

import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl

from plot_variables import M, STYLE, STYLE_M, n_for, K_for, repetitions_for, LABEL_RENAME, DATASET_RENAME


################################################################################

# 7 Main:       California Housing, Covertype, FMA, KDD-Protein, Pose, RNA, Song
# 8 Additional: Airfoil, Concrete, Banking1, Banking2, Banking3, Ijcnn1, MiniBooNE, SUN Attribute

datasets = ['BankProblem1','BankProblem2','BankProblem3',
            'airfoil','california','concrete','covertype','fma',
            'ijcnn1','kdd-protein','miniboone','pose','rna','song', 'sun-attribute']

INIT = ['Uniform','FurthestFirst','FurthestSum','AAcoreset','KMpp','AApp']

res_table = {}
for dataset in datasets:
    res_table[dataset] = {}
    for init in INIT:
        res_table[dataset][init] = {}
        for k in K_for[dataset]:
            res_table[dataset][init][k] = {}
            for metric in ['median','worst']:
                res_table[dataset][init][k][metric] = {'init':0,'overall':0}
            res_table[dataset][init][k]['median_end'] = None


preprocessing = 'CenterAndMaxScale'
# preprocessing = 'Standardize'
max_iterations = 30
for dataset in datasets:
    K = K_for[dataset]
    for l, k in enumerate(K_for[dataset]):
        for init in INIT:
            m = 0
            filename = f'result_{dataset}_{k}_{init}_{m}_{preprocessing}_{repetitions_for[dataset]}_{max_iterations}_{M}.npz'

            location = os.path.join('results', filename)
            if not os.path.exists(location):
                print(f'file not found: {location}')
                continue
            res_data = np.load(location)
            res_time_init = res_data['res_time_init']
            res_rss_init = res_data['res_rss_init'] #/ n_for[dataset]
            res_time_AA = res_data['res_time_AA']
            res_rss_AA = res_data['res_rss_AA'] #/ n_for[dataset]

            res = np.hstack((res_rss_init.reshape(-1,1), res_rss_AA))

            rss_mean = np.mean(res, axis=0)
            rss_std = np.std(res, axis=0)
            rss_stderr = rss_std / np.sqrt(res_rss_AA.shape[0])
            rss_median = np.median(res, axis=0)
            rss_quantile_upper = np.quantile(res, q=0.75, axis=0)
            rss_quantile_lower = np.quantile(res, q=0.25, axis=0)

            # worst single run
            worst_init = res_rss_init.min()
            worst_overall = res.min()
            if worst_init > res_table[dataset][init][k]['worst']['init']:
                res_table[dataset][init][k]['worst']['init'] = worst_init
            if worst_overall > res_table[dataset][init][k]['worst']['overall']:
                res_table[dataset][init][k]['worst']['overall'] = worst_overall

            # worst median
            worst_init = rss_median[0]
            worst_overall = rss_median.min()
            if worst_init > res_table[dataset][init][k]['median']['init']:
                res_table[dataset][init][k]['median']['init'] = worst_init
            if worst_overall > res_table[dataset][init][k]['median']['overall']:
                res_table[dataset][init][k]['median']['overall'] = worst_overall

            res_table[dataset][init][k]['median_end'] = rss_median[-1]




for metric in ['median','worst']:
    for dataset in datasets:
        for k in K_for[dataset]:
            for init in INIT:
                print(f'{metric} {dataset:10} {init:13} {k:3d} {res_table[dataset][init][k][metric]["init"]:6.10f} {res_table[dataset][init][k][metric]["overall"]:6.10f}')




for metric in ['median','worst']:
    for dataset in datasets:
        for k in K_for[dataset]:
            worst_init = INIT[np.argmax([res_table[dataset][init][k][metric]['init'] for init in INIT])]
            worst_overall = INIT[np.argmax([res_table[dataset][init][k][metric]['overall'] for init in INIT])]
            print(f'{metric} {dataset:10} {k:3d} init: {worst_init:13} overall: {worst_overall:13}')





table_worst_init = np.zeros((len(INIT), len(K)), dtype=np.int32)
table_worst_all = np.zeros((len(INIT), len(K)), dtype=np.int32)
table_median_init = np.zeros((len(INIT), len(K)), dtype=np.int32)
table_median_all = np.zeros((len(INIT), len(K)), dtype=np.int32)


num_datasets = 0
for dataset in datasets:
    if K_for[dataset] == [15, 25, 50, 75, 100]:
        num_datasets += 1
        for b, k in enumerate(K):
            a = np.argmax([res_table[dataset][init][k]['worst']['init'] for init in INIT])
            table_worst_init[a,b] += 1
            a = np.argmax([res_table[dataset][init][k]['worst']['overall'] for init in INIT])
            table_worst_all[a,b] += 1
            a = np.argmax([res_table[dataset][init][k]['median']['init'] for init in INIT])
            table_median_init[a,b] += 1
            a = np.argmax([res_table[dataset][init][k]['median']['overall'] for init in INIT])
            table_median_all[a,b] += 1




plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 12})

method = INIT[:]
k_names = list(map(lambda k: rf'$k$={k}', [15, 25, 50, 75, 100]))

table_ = [table_worst_init,
          table_worst_all,
          table_median_init,
          table_median_all]
title_ = ['Worst Initialization',
          'Worst Overall',
          'Median Initialization',
          'Median Overall']


fig, axv = plt.subplots(1, 4, figsize=(8, 4))
# https://stackoverflow.com/a/14779462
#cmap = plt.cm.viridis  # define the colormap
cmap = plt.cm.Reds  # define the colormap
# extract all colors from the map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
bounds = np.arange(0, len(datasets)+2, 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
for ax in axv:
    ax.set_xticks([])
    ax.set_yticks([])
axv[0].set_yticks(np.arange(len(k_names)), labels=k_names)
for ax, title, table in zip(axv, title_, table_):
    ax.set_title(title)
    ax.set_xticks(np.arange(len(INIT)), labels=list(map(lambda l: LABEL_RENAME[l], INIT)),
                  rotation=45, ha='right', rotation_mode='anchor')
    im = ax.imshow(table.T, cmap=cmap, vmin=0, vmax=len(datasets))
    # Loop over data dimensions and create text annotations.
    for i in range(len(INIT)):
        for j in range(len(k_names)):
            color = 'w'
            if table[i, j] < 5:
                color = 'k'
            text = ax.text(i, j, table[i, j], ha='center', va='center', color=color)
ax2 = fig.add_axes([1, 0.3525, 0.03, 0.295]) # (left, bottom, width, height)
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                               spacing='proportional',
                               ticks=np.arange(1,16,2).astype(np.float64)-0.5,
                               boundaries=bounds, format='%1i')

fig.tight_layout()

fig.savefig(f'/tmp/table_worst_{preprocessing}.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig(f'/tmp/table_worst_{preprocessing}.png', dpi=300, transparent=False, bbox_inches='tight')



