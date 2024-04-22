import fire
import os.path

import AApp
import utils
import baselines
import archetypalanalysis as AA

from time import time

import numpy as np


def initialize(method, X, k, m, M):
    # m is only needed for MC methods!
    ind = []
    if method == 'Uniform':
        ind = baselines.Uniform(X, k)
    elif method == 'FurthestFirst':
        ind = baselines.FurthestFirst(X, k)
    elif method == 'FurthestSum':
        ind = baselines.FurthestSum(X, k)
    elif method == 'AApp':
        ind = AApp.AApp(X, k, M)
    elif method == 'AAppMC':
        ind = AApp.AApp_MC(X, k, m, M)
    elif method == 'KMpp':
        ind = AApp.k_means_pp(X, k)
    elif method == 'KMppMC':
        ind = AApp.k_means_pp_MC(X, k, m)
    elif method == 'AAcoreset':
        ind = baselines.AAcoreset(X, k)
    else:
        raise NotImplementedError
    return ind


def main(dataset, k, init_method, m=0, preprocessing=None, repetitions=30, max_iterations=30, M=1000.0):
    # preprocessing can be
    # - None
    # - Standardize
    # - CenterAndMaxScale

    # load data
    X, y = utils.load_data(dataset, preprocessing=preprocessing)

    # within the functions used, m is a number of samples
    # but for calling this script, m is a percentage in [0,100]
    # thus, we need to compute the number of samples corresponding to that percentage
    m = int(np.ceil(X.shape[0]/100*m))


    # file name format is result_DATA_K_INIT_M_PREPROCESSING_REPS_MAXITER_M.npz
    filename = f'result_{dataset}_{k}_{init_method}_{m}_{preprocessing}_{repetitions}_{max_iterations}_{M}.npz'

    # dont run script if results file already exists!
    if os.path.exists(filename):
        print(f'ABORT: experiment {filename} already exists!')
        exit()

    # run experiment
    print(f'Processing data {dataset} with k={k} and prep.={preprocessing} (rep={repetitions} maxit={max_iterations} m={m} M={M})')

    config = {
        'dataset':dataset,
        'k':k,
        'init_method':init_method,
        'm':m,
        'preprocessing':preprocessing,
        'repetitions':repetitions,
        'max_iterations':max_iterations,
        'M':M,
        'start_time':time()
    }
    print('Config:', config)
    # save at least the config such that the file exists and no parallel run of
    # the same experiment can happen
    np.savez(filename, config=config)


    res_time_init = []
    res_rss_init = []
    res_time_AA = []
    res_rss_AA = []

    for r in range(repetitions):
        # set seed
        np.random.seed(r)

        # compute initialization
        t1 = time()
        ind = initialize(init_method, X, k, m, M)
        Z_init = X[ind].copy()
        t2 = time()
        time_init = t2-t1
        res_time_init.append(time_init)

        # compute RSS of initialization
        A = AA.ArchetypalAnalysis_compute_A(X, Z_init, M=M)
        rss_init = AA.RSS_Z(X, A, Z_init)
        res_rss_init.append(rss_init)

        # compute max_iterations iterations of Archetypal Analysis
        Z, A, B, rss_AA, time_AA = AA.ArchetypalAnalysis(X, Z_init, k, stop=False, max_iterations=max_iterations, M=M)
        res_time_AA.append(time_AA)
        res_rss_AA.append(rss_AA)

        save_time = time()

        # file name format is result_DATA_K_INIT_M_PREPROCESSING_REPS_MAXITER_M.npz
        np.savez(filename,
                 config=config, res_time_init=res_time_init, res_rss_init=res_rss_init,
                 res_time_AA=res_time_AA, res_rss_AA=np.array(res_rss_AA), r=r, save_time=save_time)


if __name__ == '__main__':
  fire.Fire(main)

