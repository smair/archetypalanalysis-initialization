import re
import os.path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file, fetch_california_housing
from scipy.io import loadmat

data_path = '/home/sebastian/data/'


def load_data(dataset, preprocessing=None):
    X = []
    y = []

    if dataset == 'covertype':  # (581012, 54)
        # Forest cover type
        # https://archive.ics.uci.edu/ml/datasets/covertype
        X, y = load_svmlight_file(data_path + 'covtype.libsvm.binary')
        X = np.asarray(X.todense())
    elif dataset == 'ijcnn1':  # (49990, 22)
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
        X, y = load_svmlight_file(data_path + "ijcnn1/ijcnn1")
        X = np.asarray(X.todense())
    elif dataset == 'song':  # (515345, 90)
        # YearPredictionMSD is a subset of the Million Song Dataset
        # https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
        data = np.loadtxt(
            data_path + 'YearPredictionMSD.txt', skiprows=0, delimiter=','
        )
        X = data[:, 1:]
        y = data[:, 0]
    elif dataset == 'pose':  # (35832, 48)
        # ECCV 2018 PoseTrack Challenge
        # http://vision.imar.ro/human3.6m/challenge_open.php
        X = []
        cache_file = data_path + 'Human3.6M/ECCV18_Challenge/train_cache.npz'
        if os.path.exists(cache_file):
            print('utils.load_data(): loading cache file for pose data')
            npz = np.load(cache_file)
            X = npz['X']
        else:
            X = []
            for i in tqdm(range(1, 35832 + 1), desc='loading pose'):
                f = data_path + 'Human3.6M/ECCV18_Challenge/Train/POSE/{:05d}.csv'.format(i)
                data = np.loadtxt(f, skiprows=0, delimiter=",")
                X.append(data[1:, :].flatten())
            X = np.array(X)
            print('utils.load_data(): saving cache file for pose data')
            np.savez(cache_file, X=X)
    elif dataset == 'kdd-protein': # (145751, 74)
        # KDD Cup 2004
        # http://osmot.cs.cornell.edu/kddcup/datasets.html
        #
        # Protein Homology Dataset
        #
        # Example:
        # 279 261532 0 52.00 32.69 ... -0.350 0.26 0.76
        #
        # 279 is the BLOCK ID.
        # 261532 is the EXAMPLE ID.
        # The "0" in the third column is the target value. This indicates that this
        #      protein is not homologous to the native sequence (it is a decoy).
        #      If this protein was homologous the target would be "1".
        # Columns 4-77 are the input attributes.

        data = pd.read_csv(data_path+'KDD_protein/bio_train.dat', sep='\t', skiprows=0, header=None)
        X = np.asarray(data)[:,3:]
    elif dataset == 'rna': # (488565, 8)
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
        X_train, y_train = load_svmlight_file(data_path+'RNA/libsvmtools_datasets/cod-rna')
        X_train = np.asarray(X_train.todense())
        X_val, y_val = load_svmlight_file(data_path+'RNA/libsvmtools_datasets/cod-rna.t')
        X_val = np.asarray(X_val.todense())
        X_rest, y_rest = load_svmlight_file(data_path+'RNA/libsvmtools_datasets/cod-rna.r')
        X_rest = np.asarray(X_rest.todense())

        X = np.vstack((X_train,X_val,X_rest))
        y = np.hstack((y_train,y_val,y_rest))
    elif dataset == 'miniboone': # (130064, 50)
        # https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification
        data = pd.read_csv(data_path+'MiniBooNE_PID.txt', sep='\s\s*', skiprows=[0], header=None, engine='python')
        X = np.asarray(data)
    elif dataset == 'california':
        X, y = fetch_california_housing(data_home=data_path, download_if_missing=True, return_X_y=True)
    elif dataset == 'airfoil':
        # https://archive.ics.uci.edu/dataset/291/airfoil+self+noise
        f = data_path + 'airfoil_self_noise.dat'
        data = pd.read_csv(f, sep="\t", header=None)
        X = np.asarray(data)[:, :-1]
        y = np.asarray(data)[:, -1]
    elif dataset == 'concrete':
        # https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
        f = data_path + 'Concrete_Data.xls'
        data = pd.read_excel(f)
        X = np.asarray(data)[:, :-1]
        y = np.asarray(data)[:, -1]
    elif re.match('^BankProblem\d$', dataset) is not None:
        # https://www.people.vcu.edu/~jdula/FramesAlgorithms/BankingData/
        data = np.loadtxt(data_path+'BankingData/'+dataset+'.con',comments='%',skiprows=12)
        X = np.asarray(data)
    elif dataset == 'sun-attribute':
        # https://cs.brown.edu/~gmpatter/sunattributes.html
        X = loadmat(data_path+'SUNAttributeDB/attributeLabels_continuous.mat')['labels_cv']
    elif dataset == 'fma':
        # https://github.com/mdeff/fma
        data = pd.read_csv(data_path+'fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
        X = np.asarray(data)
    else:
        raise NotImplementedError

    if preprocessing == 'Standardize':
        X = StandardScaler().fit_transform(X)
    elif preprocessing == 'CenterAndMaxScale':
        X = X - X.mean(0)
        X = X / X.max()

    return X, y



