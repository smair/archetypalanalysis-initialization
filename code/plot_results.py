import os.path

import numpy as np
import matplotlib.pylab as plt

from plot_variables import M, STYLE, STYLE_M, n_for, K_for, repetitions_for, time_normalizer_for, LABEL_RENAME, DATASET_RENAME

################################################################################
# main results with iterations
# These are Figures 4 and 12 of the paper.

datasets = ['california','covertype','fma','kdd-protein','pose','rna','song']
INIT = ['Uniform','FurthestFirst','FurthestSum','AAcoreset','KMpp','AApp','AAppMC']
m_ = [20,10,5,1]
for j, m in enumerate(m_):
    STYLE[f'AAppMC{m}'] = STYLE['AAppMC'].copy()
    STYLE[f'AAppMC{m}']['ls'] = STYLE_M[j]
STYLE['AAppMC1']['color']  = 'cyan'

INIT_ = INIT[:-1]+list(map(lambda x: x[0]+str(x[1]), list(zip(('AAppMC',)*len(m_), m_))))

K = K_for['pose']

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 12})


zoom = {'main':{'CenterAndMaxScale':{'covertype':{15:[25, 30, 4.2e-4, 6.9e-4]
                                                  },
                                     'fma':{15:[25, 30, 3.5e-4, 8.2e-4],
                                            25:[25, 30, 2.5e-4, 4e-4],
                                            50:[25, 30, 1.25e-4, 3.4e-4],
                                            75:[25, 30, 1.2e-4, 2.1e-4],
                                            100:[25, 30, 1e-4, 2.1e-4],
                                            },
                                     'pose':{15:[25, 30, 2.9e-1, 2.97e-1],
                                             },
                                     'song':{15:[25, 30, 3.5e-4, 4.5e-4],
                                             25:[25, 30, 2.85e-4, 3.05e-4],
                                             50:[25, 30, 2.05e-4, 2.155e-4],
                                             75:[25, 30, 1.65e-4, 1.75e-4],
                                             100:[25, 30, 1.425e-4, 1.52e-4],
                                             }
                                     },
                'Standardize':{'covertype':{15:[25, 30, 3.3e1, 3.9e1],
                                            25:[25, 30, 2.4e1, 2.6e1],
                                            50:[25, 30, 5.5, 6.9],
                                            75:[25, 30, 3.5, 5],
                                            100:[25, 30, 2.75, 4]
                                           },
                               'fma':{15:[25, 30, 3.2e2, 3.85e2],
                                      25:[25, 30, 2.9e2, 3.5e2],
                                      50:[25, 30, 2.6e2, 3.0e2],
                                      75:[25, 30, 2.35e2, 2.7e2],
                                      100:[25, 30, 2.25e2, 2.5e2],
                                      },
                               'kdd-protein':{15:[25, 30, 3.25e1, 3.35e1],
                                              25:[25, 30, 2.875e1, 2.975e1],
                                              50:[25, 30, 2.4e1, 2.54e1],
                                              },
                               'pose':{15:[25, 30, 11, 12],
                                       25:[25, 30, 8.4, 8.75],
                                       50:[25, 30, 5.975, 6.1],
                                       75:[25, 30, 5.00, 5.2],
                                       100:[25, 30, 4.4, 4.6],
                                       },
                               'song':{15:[25, 30, 5.05e1, 5.6e1],
                                       25:[25, 30, 4.4e1, 4.475e1],
                                       50:[25, 30, 3.46e1, 3.55e1],
                                       75:[25, 30, 2.95e1, 3.025e1],
                                       100:[25, 30, 2.6e1, 2.68e1]
                                       }
                               }
                }
        }

max_iterations = 30
show_seeds = False
show_median = True
for preprocessing in ['CenterAndMaxScale', 'Standardize']:
    height = 16
    if preprocessing == 'Standardize':
        height = 19
    fig, axv = plt.subplots(len(datasets), len(K), figsize=(16,height), sharex=True) # height was 13
    for row, dataset in enumerate(datasets):
        dom = np.hstack((-0.25, 0.25, np.arange(1, max_iterations+1)))
        K = K_for[dataset]
        for col, k in enumerate(K_for[dataset]):
            ax = axv[row,col]
            ax.set_title(fr'{DATASET_RENAME[dataset]} $k$={k}')
            if row == len(datasets)-1:
                ax.set_xlabel(r'Iterations of AA')
            if col == 0:
                ax.set_ylabel(r'MSE')
            ax.set_yscale('log')
            ax.set_xticks(np.arange(0, max_iterations+1, 5))
            ax.set_xticklabels(['init.']+list(np.arange(5, max_iterations+1, 5)))

            # zoom
            if 'main' in zoom and preprocessing in zoom['main'] and dataset in zoom['main'][preprocessing] and k in zoom['main'][preprocessing][dataset]:
                axins = ax.inset_axes([0.3, 0.35, 0.6, 0.6]) # [x0, y0, width, height]

            for init in INIT_:
                m = 0
                filename = f'result_{dataset}_{k}_{init}_{m}_{preprocessing}_{repetitions_for[dataset]}_{max_iterations}_{M}.npz'

                if 'MC' in init:
                    tmp = init.split('MC')
                    init_ = tmp[0]+'MC'
                    m_percent = int(tmp[1])
                    if n_for[dataset] < 25000 and m_percent == 1:
                        # skip the 1% MCMC approximation if the data set is too small
                        continue
                    # convert a percentage to a chain length
                    m = int(np.ceil(n_for[dataset]/100*m_percent))
                    filename = f'result_{dataset}_{k}_{init_}_{m}_{preprocessing}_{repetitions_for[dataset]}_{max_iterations}_{M}.npz'

                location = os.path.join('results', filename)
                if not os.path.exists(location):
                    print(f'file not found: {location}')
                    continue
                res_data = np.load(location)
                if len(res_data.keys()) == 1:
                    print(f'run not finished: {location}')
                    continue
                res_time_init = res_data['res_time_init']
                res_rss_init = res_data['res_rss_init'] / n_for[dataset]
                res_time_AA = res_data['res_time_AA']
                res_rss_AA = res_data['res_rss_AA'] / n_for[dataset]

                if res_data['res_rss_AA'].shape[0] != repetitions_for[dataset]:
                    print(f'run not complete: {location}')
                    #continue

                res = np.hstack((res_rss_init.reshape(-1,1), res_rss_init.reshape(-1,1), res_rss_AA))

                rss_mean = np.mean(res, axis=0)
                rss_std = np.std(res, axis=0)
                rss_stderr = rss_std / np.sqrt(res_rss_AA.shape[0])
                rss_median = np.median(res, axis=0)
                rss_quantile_upper = np.quantile(res, q=0.75, axis=0)
                rss_quantile_lower = np.quantile(res, q=0.25, axis=0)

                rss_upper = None
                rss_avg = None
                rss_lower = None
                if show_median:
                    rss_upper = rss_quantile_upper
                    rss_avg = rss_median
                    rss_lower = rss_quantile_lower
                else: # show mean
                    rss_upper = rss_mean+rss_std
                    rss_avg = rss_mean
                    rss_lower = rss_mean-rss_std

                ax.fill_between(dom, rss_lower, rss_upper,
                                alpha=0.2, **STYLE[init])
                if show_seeds:
                    for i in range(repetitions_for[dataset]):
                        ax.plot(dom, res[i], alpha=0.2, **STYLE[init])
                ax.plot(dom, rss_avg, label=LABEL_RENAME[init], **STYLE[init])

                # zoom
                if 'main' in zoom and preprocessing in zoom['main'] and dataset in zoom['main'][preprocessing] and k in zoom['main'][preprocessing][dataset]:
                    axins.plot(dom, rss_avg, label=LABEL_RENAME[init], **STYLE[init])

            # zoom
            if 'main' in zoom and preprocessing in zoom['main'] and dataset in zoom['main'][preprocessing] and k in zoom['main'][preprocessing][dataset]:
                x1, x2, y1, y2 = zoom['main'][preprocessing][dataset][k]
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                ax.indicate_inset_zoom(axins, edgecolor="black")

            ax.grid(True, which='both')

    fig.tight_layout()

    # https://stackoverflow.com/questions/45597092/expanded-legend-over-2-subplots
    s = fig.subplotpars
    bb = [s.left, s.top+0.025, s.right-s.left, 0.05]
    ax.legend(bbox_to_anchor=bb, loc='lower left',
              bbox_transform=fig.transFigure,
              mode='expand', borderaxespad=0, ncol=10)

    fig.savefig(f'/tmp/result_main_iterations_{preprocessing}_{show_median}_larger.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'/tmp/result_main_iterations_{preprocessing}_{show_median}_larger.png', dpi=300, transparent=False, bbox_inches='tight')



################################################################################
# additional results with iterations
# These are Figures 7 and 14 of the paper.

datasets = ['airfoil','BankProblem1','BankProblem2','BankProblem3',
            'concrete','ijcnn1','miniboone','sun-attribute']
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


zoom = {'additional':{'CenterAndMaxScale':{'concrete':{15:[25, 30, 1.2e-2, 1.5e-2]
                                                       },
                                           'ijcnn1':{15:[25, 30, 2.62e-1, 2.65e-1],
                                                     25:[25, 30, 1.45e-1, 1.75e-1],
                                                     50:[25, 30, 9.5e-2, 1.1e-1],
                                                     75:[25, 30, 7.1e-2, 7.8e-2],
                                                     100:[25, 30, 5.5e-2, 6.225e-2],
                                                     },
                                           'sun-attribute':{15:[25, 30, 2.05, 2.085],
                                                            25:[25, 30, 1.84, 1.9],
                                                            50:[25, 30, 1.6, 1.7],
                                                            75:[25, 30, 1.50, 1.51],
                                                            100:[25, 30, 1.421, 1.43],
                                                            },
                                           },
                      'Standardize':{'ijcnn1':{15:[25, 30, 6.3, 6.525],
                                               25:[25, 30, 3.55, 4],
                                               50:[25, 30, 2.4, 2.48],
                                               75:[25, 30, 1.775, 1.9],
                                               100:[25, 30, 1.4, 1.54],
                                               },
                                     'sun-attribute':{15:[25, 30, 61.5, 64],
                                                      25:[25, 30, 53.5, 54.4],
                                                      50:[25, 30, 41.75, 44],
                                                      75:[25, 30, 38, 40.5],
                                                      100:[25, 30, 36, 39],
                                                      }
                                     }
                      }
        }

max_iterations = 30
show_seeds = False
show_median = True
for preprocessing in ['CenterAndMaxScale', 'Standardize']:
    fig, axv = plt.subplots(len(datasets), len(K), figsize=(16,19), sharex=True)
    for row, dataset in enumerate(datasets):
        dom = np.hstack((-0.25, 0.25, np.arange(1, max_iterations+1)))
        K = K_for[dataset]
        for col, k in enumerate(K_for[dataset]):
            ax = axv[row,col]
            ax.set_title(fr'{DATASET_RENAME[dataset]} $k$={k}')
            if row == len(datasets)-1:
                ax.set_xlabel(r'Iterations of AA')
            if col == 0:
                ax.set_ylabel(r'MSE')
            ax.set_yscale('log')
            ax.set_xticks(np.arange(0, max_iterations+1, 5))
            ax.set_xticklabels(['init.']+list(np.arange(5, max_iterations+1, 5)))

            # zoom
            if 'additional' in zoom and preprocessing in zoom['additional'] and dataset in zoom['additional'][preprocessing] and k in zoom['additional'][preprocessing][dataset]:
                axins = ax.inset_axes([0.3, 0.35, 0.6, 0.6]) # [x0, y0, width, height]

            for init in INIT_:
                m = 0
                filename = f'result_{dataset}_{k}_{init}_{m}_{preprocessing}_{repetitions_for[dataset]}_{max_iterations}_{M}.npz'

                if 'MC' in init:
                    tmp = init.split('MC')
                    init_ = tmp[0]+'MC'
                    m_percent = int(tmp[1])
                    if n_for[dataset] < 25000 and m_percent == 1:
                        # skip the 1% MCMC approximation if the data set is too small
                        continue
                    # convert a percentage to a chain length
                    m = int(np.ceil(n_for[dataset]/100*m_percent))
                    filename = f'result_{dataset}_{k}_{init_}_{m}_{preprocessing}_{repetitions_for[dataset]}_{max_iterations}_{M}.npz'

                location = os.path.join('results', filename)
                if not os.path.exists(location):
                    print(f'file not found: {location}')
                    continue
                res_data = np.load(location)
                if len(res_data.keys()) == 1:
                    print(f'run not finished: {location}')
                    continue
                res_time_init = res_data['res_time_init']
                res_rss_init = res_data['res_rss_init'] / n_for[dataset]
                res_time_AA = res_data['res_time_AA']
                res_rss_AA = res_data['res_rss_AA'] / n_for[dataset]

                if res_data['res_rss_AA'].shape[0] != repetitions_for[dataset]:
                    print(f'run not complete: {location}')
                    #continue

                res = np.hstack((res_rss_init.reshape(-1,1), res_rss_init.reshape(-1,1), res_rss_AA))

                rss_mean = np.mean(res, axis=0)
                rss_std = np.std(res, axis=0)
                rss_stderr = rss_std / np.sqrt(res_rss_AA.shape[0])
                rss_median = np.median(res, axis=0)
                rss_quantile_upper = np.quantile(res, q=0.75, axis=0)
                rss_quantile_lower = np.quantile(res, q=0.25, axis=0)

                rss_upper = None
                rss_avg = None
                rss_lower = None
                if show_median:
                    rss_upper = rss_quantile_upper
                    rss_avg = rss_median
                    rss_lower = rss_quantile_lower
                else: # show mean
                    rss_upper = rss_mean+rss_std
                    rss_avg = rss_mean
                    rss_lower = rss_mean-rss_std

                ax.fill_between(dom, rss_lower, rss_upper,
                                alpha=0.2, **STYLE[init])
                if show_seeds:
                    for i in range(repetitions_for[dataset]):
                        ax.plot(dom, res[i], alpha=0.2, **STYLE[init])
                ax.plot(dom, rss_avg, label=LABEL_RENAME[init], **STYLE[init])

                # zoom
                if 'additional' in zoom and preprocessing in zoom['additional'] and dataset in zoom['additional'][preprocessing] and k in zoom['additional'][preprocessing][dataset]:
                    axins.plot(dom, rss_avg, label=LABEL_RENAME[init], **STYLE[init])

            # zoom
            if 'additional' in zoom and preprocessing in zoom['additional'] and dataset in zoom['additional'][preprocessing] and k in zoom['additional'][preprocessing][dataset]:
                x1, x2, y1, y2 = zoom['additional'][preprocessing][dataset][k]
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                ax.indicate_inset_zoom(axins, edgecolor="black")

            ax.grid(True, which='both')

    fig.tight_layout()

    # https://stackoverflow.com/questions/45597092/expanded-legend-over-2-subplots
    s = fig.subplotpars
    bb = [s.left, s.top+0.025, s.right-s.left, 0.05]
    ax.legend(bbox_to_anchor=bb, loc='lower left',
              bbox_transform=fig.transFigure,
              mode='expand', borderaxespad=0, ncol=10)

    fig.savefig(f'/tmp/result_additional_iterations_{preprocessing}_{show_median}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'/tmp/result_additional_iterations_{preprocessing}_{show_median}.png', dpi=300, transparent=False, bbox_inches='tight')


################################################################################
# main results with time
# These are Figures 10 and 11 of the paper.

datasets = ['california','covertype','fma','kdd-protein','pose','rna','song']
INIT = ['Uniform','FurthestFirst','FurthestSum','AAcoreset','KMpp','AApp','AAppMC']
m_ = [20,10,5,1]
for j, m in enumerate(m_):
    STYLE[f'AAppMC{m}'] = STYLE['AAppMC'].copy()
    STYLE[f'AAppMC{m}']['ls'] = STYLE_M[j]
STYLE['AAppMC1']['color']  = 'cyan'

INIT_ = INIT[:-1]+list(map(lambda x: x[0]+str(x[1]), list(zip(('AAppMC',)*len(m_), m_))))

K = K_for['pose']

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 12})

zoom = {'main':{'CenterAndMaxScale':{'fma':{15:[18, 40, 3.5e-4, 8.2e-4],
                                            25:[25, 65, 2.5e-4, 4e-4],
                                            50:[75, 165, 1.25e-4, 3.4e-4],
                                            75:[165, 295, 1.2e-4, 2.1e-4],
                                            100:[220, 505, 1e-4, 2.1e-4]
                                            },
                                     'pose':{15:[70, 100, 2.9e-1, 2.97e-1],
                                             },
                                     'song':{15:[25, 60, 3.5e-4, 4.5e-4],
                                             25:[50, 80, 2.85e-4, 3.05e-4],
                                             50:[90, 150, 2.05e-4, 2.155e-4],
                                             75:[140, 285, 1.65e-4, 1.75e-4],
                                             100:[210, 410, 1.425e-4, 1.52e-4],
                                             }
                                     },
                'Standardize':{'covertype':{15:[50, 120, 3.3e1, 3.9e1],
                                            25:[60, 108, 2.4e1, 2.6e1],
                                            50:[90, 135, 5.5, 6.9],
                                            75:[90, 200, 3.5, 5],
                                            100:[90, 275, 2.75, 4]
                                           },
                               'fma':{15:[30, 60, 3.2e2, 3.85e2],
                                      25:[45, 95, 2.9e2, 3.5e2],
                                      50:[50, 180, 2.6e2, 3.0e2],
                                      75:[150, 300, 2.35e2, 2.7e2],
                                      100:[155, 450, 2.25e2, 2.5e2]
                                      },
                               'kdd-protein':{15:[7.25, 9, 3.25e1, 3.35e1],
                                              25:[9, 13, 2.875e1, 2.975e1],
                                              50:[13, 29, 2.4e1, 2.54e1],
                                              },
                               'pose':{15:[69, 98, 11, 12],
                                       25:[90, 150, 8.4, 8.75],
                                       50:[140, 280, 5.975, 6.1],
                                       75:[240, 475, 5.00, 5.2],
                                       100:[280, 570, 4.4, 4.6],
                                       },
                               'song':{15:[40, 69, 5.05e1, 5.6e1],
                                       25:[50, 87, 4.4e1, 4.475e1],
                                       50:[90, 142, 3.46e1, 3.55e1],
                                       75:[120, 240, 2.95e1, 3.025e1],
                                       100:[180, 390, 2.6e1, 2.68e1]
                                       }
                               }
                }
        }

max_iterations = 30
show_seeds = False
show_median = True
for preprocessing in ['CenterAndMaxScale', 'Standardize']:
    fig, axv = plt.subplots(len(datasets), len(K), figsize=(16,19)) # height was 14
    for row, dataset in enumerate(datasets):
        K = K_for[dataset]
        for col, k in enumerate(K_for[dataset]):
            ax = axv[row,col]
            ax.set_title(fr'{DATASET_RENAME[dataset]} $k$={k}')
            if col == 0:
                ax.set_ylabel(r'MSE')
            ax.set_yscale('log')

            # zoom
            if 'main' in zoom and preprocessing in zoom['main'] and dataset in zoom['main'][preprocessing] and k in zoom['main'][preprocessing][dataset]:
                axins = ax.inset_axes([0.3, 0.35, 0.6, 0.6]) # [x0, y0, width, height]

            for init in INIT_:
                m = 0
                filename = f'result_{dataset}_{k}_{init}_{m}_{preprocessing}_{repetitions_for[dataset]}_{max_iterations}_{M}.npz'

                if 'MC' in init:
                    tmp = init.split('MC')
                    init_ = tmp[0]+'MC'
                    m_percent = int(tmp[1])
                    if n_for[dataset] < 25000 and m_percent == 1:
                        # skip the 1% MCMC approximation if the data set is too small
                        continue
                    # convert a percentage to a chain length
                    m = int(np.ceil(n_for[dataset]/100*m_percent))
                    filename = f'result_{dataset}_{k}_{init_}_{m}_{preprocessing}_{repetitions_for[dataset]}_{max_iterations}_{M}.npz'

                location = os.path.join('results', filename)
                if not os.path.exists(location):
                    print(f'file not found: {location}')
                    continue
                res_data = np.load(location)
                if len(res_data.keys()) == 1:
                    print(f'run not finished: {location}')
                    continue
                res_time_init = res_data['res_time_init'].reshape(-1,1)
                res_rss_init = res_data['res_rss_init'].reshape(-1,1) / n_for[dataset]
                res_time_AA = res_data['res_time_AA']
                res_rss_AA = res_data['res_rss_AA'] / n_for[dataset]

                if res_data['res_rss_AA'].shape[0] != repetitions_for[dataset]:
                    print(f'run not complete: {location}')
                    #continue

                if time_normalizer_for[dataset] == 60.0:
                    ax.set_xlabel('Computation time in minutes')
                else:
                    ax.set_xlabel('Computation time in seconds')

                res_time = np.hstack((res_time_init, res_time_AA+res_time_init)) / time_normalizer_for[dataset]
                res_rss = np.hstack((res_rss_init, res_rss_AA))

                # res_time_mean = np.mean(res_time, axis=0)
                # res_time_std = np.std(res_time, axis=0)
                # res_time_stderr = rss_std / np.sqrt(res_time.shape[0])
                res_time_median = np.median(res_time, axis=0)
                # res_time_quantile_upper = np.quantile(res_time, q=0.75, axis=0)
                # res_time_quantile_lower = np.quantile(res_time, q=0.25, axis=0)

                res_rss_mean = np.mean(res_rss, axis=0)
                res_rss_std = np.std(res_rss, axis=0)
                # res_rss_stderr = rss_std / np.sqrt(res_rss.shape[0])
                res_rss_median = np.median(res_rss, axis=0)
                res_rss_quantile_upper = np.quantile(res_rss, q=0.75, axis=0)
                res_rss_quantile_lower = np.quantile(res_rss, q=0.25, axis=0)

                res_rss_upper = None
                res_rss_avg = None
                res_rss_lower = None
                if show_median:
                    res_rss_upper = res_rss_quantile_upper
                    res_rss_avg = res_rss_median
                    res_rss_lower = res_rss_quantile_lower
                else: # show mean
                    res_rss_upper = res_rss_mean+res_rss_std
                    res_rss_avg = res_rss_mean
                    res_rss_lower = res_rss_mean-res_rss_std

                ax.fill_between(res_time_median, res_rss_lower, res_rss_upper,
                                alpha=0.2, **STYLE[init])
                if show_seeds:
                    for i in range(repetitions_for[dataset]):
                        ax.plot(res_time[i],
                                res_rss[i],
                                alpha=0.2, **STYLE[init])

                ax.plot(res_time_median, res_rss_avg,
                        label=LABEL_RENAME[init], **STYLE[init])

                # zoom
                if 'main' in zoom and preprocessing in zoom['main'] and dataset in zoom['main'][preprocessing] and k in zoom['main'][preprocessing][dataset]:
                    axins.plot(res_time_median, res_rss_avg, label=LABEL_RENAME[init], **STYLE[init])

                ax.locator_params(axis='x', nbins=5)

            # zoom
            if 'main' in zoom and preprocessing in zoom['main'] and dataset in zoom['main'][preprocessing] and k in zoom['main'][preprocessing][dataset]:
                x1, x2, y1, y2 = zoom['main'][preprocessing][dataset][k]
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                ax.indicate_inset_zoom(axins, edgecolor="black")

            ax.grid(True, which='both')

    fig.tight_layout()

    # https://stackoverflow.com/questions/45597092/expanded-legend-over-2-subplots
    s = fig.subplotpars
    bb = [s.left, s.top+0.025, s.right-s.left, 0.05]
    ax = axv[0,0]
    ax.legend(bbox_to_anchor=bb, loc='lower left',
              bbox_transform=fig.transFigure,
              mode='expand', borderaxespad=0, ncol=9)

    fig.savefig(f'/tmp/result_main_time_{preprocessing}_{show_median}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'/tmp/result_main_time_{preprocessing}_{show_median}.png', dpi=300, transparent=False, bbox_inches='tight')


################################################################################
# additional results with time
# These are Figures 13 and 15 of the paper.

datasets = ['airfoil','BankProblem1','BankProblem2','BankProblem3',
            'concrete','ijcnn1','miniboone','sun-attribute']
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

zoom = {'additional':{'CenterAndMaxScale':{'concrete':{15:[2, 3, 1.2e-2, 1.5e-2]
                                                       },
                                           'ijcnn1':{15:[75, 120, 2.625e-1, 2.645e-1],
                                                     25:[90, 160, 1.45e-1, 1.75e-1],
                                                     50:[125, 250, 9.5e-2, 1.08e-1],
                                                     75:[165, 365, 7.1e-2, 7.8e-2],
                                                     100:[200, 400, 5.5e-2, 6.225e-2],
                                                     },
                                           'sun-attribute':{15:[65, 89, 2.05, 2.085],
                                                            25:[80, 120, 1.84, 1.9],
                                                            50:[100, 210, 1.6, 1.7],
                                                            75:[150, 330, 1.50, 1.51],
                                                            100:[180, 375, 1.421, 1.43],
                                                            },
                                           },
                      'Standardize':{'ijcnn1':{15:[75, 115, 6.3, 6.525],
                                               25:[80, 140, 3.55, 4],
                                               50:[115, 235, 2.4, 2.48],
                                               75:[170, 355, 1.775, 1.9],
                                               100:[220, 395, 1.4, 1.54],
                                               },
                                     'sun-attribute':{15:[60, 100, 61.55, 64],
                                                      25:[75, 130, 53.5, 54.4],
                                                      50:[150, 255, 41.78, 44],
                                                      75:[190, 385, 38, 40.5],
                                                      100:[240, 400, 36, 39],
                                                      }
                                     }
                      }
        }

max_iterations = 30
show_seeds = False
show_median = True
for preprocessing in ['CenterAndMaxScale', 'Standardize']:
    fig, axv = plt.subplots(len(datasets), len(K), figsize=(16,19))
    for row, dataset in enumerate(datasets):
        dom = np.hstack((-0.25, 0.25, np.arange(1, max_iterations+1)))
        K = K_for[dataset]
        for col, k in enumerate(K_for[dataset]):
            ax = axv[row,col]
            ax.set_title(fr'{DATASET_RENAME[dataset]} $k$={k}')
            # if row == len(datasets)-1:
            #     ax.set_xlabel(r'Iterations of AA')
            if col == 0:
                ax.set_ylabel(r'MSE')
            ax.set_yscale('log')
            # ax.set_xticks(list(range(0, max_iterations+1)))
            # ax.set_xticklabels(['init.']+list(range(1,max_iterations+1)))

            # zoom
            if 'additional' in zoom and preprocessing in zoom['additional'] and dataset in zoom['additional'][preprocessing] and k in zoom['additional'][preprocessing][dataset]:
                axins = ax.inset_axes([0.3, 0.35, 0.6, 0.6]) # [x0, y0, width, height]

            for init in INIT_:
                m = 0
                filename = f'result_{dataset}_{k}_{init}_{m}_{preprocessing}_{repetitions_for[dataset]}_{max_iterations}_{M}.npz'

                if 'MC' in init:
                    tmp = init.split('MC')
                    init_ = tmp[0]+'MC'
                    m_percent = int(tmp[1])
                    if n_for[dataset] < 25000 and m_percent == 1:
                        # skip the 1% MCMC approximation if the data set is too small
                        continue
                    # convert a percentage to a chain length
                    m = int(np.ceil(n_for[dataset]/100*m_percent))
                    filename = f'result_{dataset}_{k}_{init_}_{m}_{preprocessing}_{repetitions_for[dataset]}_{max_iterations}_{M}.npz'

                location = os.path.join('results', filename)
                if not os.path.exists(location):
                    print(f'file not found: {location}')
                    continue
                res_data = np.load(location)
                if len(res_data.keys()) == 1:
                    print(f'run not finished: {location}')
                    continue
                res_time_init = res_data['res_time_init'].reshape(-1,1)
                res_rss_init = res_data['res_rss_init'].reshape(-1,1) / n_for[dataset]
                res_time_AA = res_data['res_time_AA']
                res_rss_AA = res_data['res_rss_AA'] / n_for[dataset]

                if res_data['res_rss_AA'].shape[0] != repetitions_for[dataset]:
                    print(f'run not complete: {location}')
                    #continue

                if time_normalizer_for[dataset] == 60.0:
                    ax.set_xlabel('Computation time in minutes')
                else:
                    ax.set_xlabel('Computation time in seconds')

                res_time = np.hstack((res_time_init, res_time_AA+res_time_init)) / time_normalizer_for[dataset]
                res_rss = np.hstack((res_rss_init, res_rss_AA))

                # res_time_mean = np.mean(res_time, axis=0)
                # res_time_std = np.std(res_time, axis=0)
                # res_time_stderr = rss_std / np.sqrt(res_time.shape[0])
                res_time_median = np.median(res_time, axis=0)
                # res_time_quantile_upper = np.quantile(res_time, q=0.75, axis=0)
                # res_time_quantile_lower = np.quantile(res_time, q=0.25, axis=0)

                res_rss_mean = np.mean(res_rss, axis=0)
                res_rss_std = np.std(res_rss, axis=0)
                # res_rss_stderr = rss_std / np.sqrt(res_rss.shape[0])
                res_rss_median = np.median(res_rss, axis=0)
                res_rss_quantile_upper = np.quantile(res_rss, q=0.75, axis=0)
                res_rss_quantile_lower = np.quantile(res_rss, q=0.25, axis=0)

                res_rss_upper = None
                res_rss_avg = None
                res_rss_lower = None
                if show_median:
                    res_rss_upper = res_rss_quantile_upper
                    res_rss_avg = res_rss_median
                    res_rss_lower = res_rss_quantile_lower
                else: # show mean
                    res_rss_upper = res_rss_mean+res_rss_std
                    res_rss_avg = res_rss_mean
                    res_rss_lower = res_rss_mean-res_rss_std

                ax.fill_between(res_time_median, res_rss_lower, res_rss_upper,
                                alpha=0.2, **STYLE[init])
                if show_seeds:
                    for i in range(repetitions_for[dataset]):
                        ax.plot(res_time[i],
                                res_rss[i],
                                alpha=0.2, **STYLE[init])

                ax.plot(res_time_median, res_rss_avg,
                        label=LABEL_RENAME[init], **STYLE[init])

                # zoom
                if 'additional' in zoom and preprocessing in zoom['additional'] and dataset in zoom['additional'][preprocessing] and k in zoom['additional'][preprocessing][dataset]:
                    axins.plot(res_time_median, res_rss_avg, label=LABEL_RENAME[init], **STYLE[init])

                ax.locator_params(axis='x', nbins=5)

            # zoom
            if 'additional' in zoom and preprocessing in zoom['additional'] and dataset in zoom['additional'][preprocessing] and k in zoom['additional'][preprocessing][dataset]:
                x1, x2, y1, y2 = zoom['additional'][preprocessing][dataset][k]
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                ax.indicate_inset_zoom(axins, edgecolor="black")

            ax.grid(True, which='both')

    fig.tight_layout()

    # https://stackoverflow.com/questions/45597092/expanded-legend-over-2-subplots
    s = fig.subplotpars
    bb = [s.left, s.top+0.025, s.right-s.left, 0.05]
    ax = axv[-1,0]
    ax.legend(bbox_to_anchor=bb, loc='lower left',
              bbox_transform=fig.transFigure,
              mode='expand', borderaxespad=0, ncol=9)


    fig.savefig(f'/tmp/result_additional_time_{preprocessing}_{show_median}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'/tmp/result_additional_time_{preprocessing}_{show_median}.png', dpi=300, transparent=False, bbox_inches='tight')

