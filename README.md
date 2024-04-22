# Archetypal Analysis++: Rethinking the Initialization Strategy

This repository contains the source code of the paper [**Archetypal Analysis++: Rethinking the Initialization Strategy**](https://openreview.net/pdf?id=KVUtlM60HM) which was reviewed on [OpenReview](https://openreview.net/forum?id=KVUtlM60HM) and published in the [Transactions on Machine Learning Research](https://jmlr.org/tmlr/).

## Abstract
Archetypal analysis is a matrix factorization method with convexity constraints. Due to local minima, a good initialization is essential, but frequently used initialization methods yield either sub-optimal starting points or are prone to get stuck in poor local minima. In this paper, we propose archetypal analysis++ (AA++), a probabilistic initialization strategy for archetypal analysis that sequentially samples points based on their influence on the objective function, similar to k-means++. In fact, we argue that k-means++ already approximates the proposed initialization method. Furthermore, we suggest to adapt an efficient Monte Carlo approximation of k-means++ to AA++. In an extensive empirical evaluation of 15 real-world data sets of varying sizes and dimensionalities and considering two pre-processing strategies, we show that AA++ almost always outperforms all baselines, including the most frequently used ones.

## Code

The code was tested with the following versions:

* python 3.9.0
* numpy 1.23.2
* scipy 1.9.0
* sklearn 1.1.1
* fire 0.4.0
* matplotlib 3.5.3
* pandas 1.4.4
* shapely 1.8.4
* tqdm 4.64.1

### Files

- `AApp.py` contains the implementation of AA++ and its approximations.
- `archetypalanalysis.py` contains the implementation of archetypal analysis.
- `baselines.py` contains the implementation of the baselines *Uniform*, *FurthestFirst*, *FurthestSum*, and *AAcoreset*.
- `build.sh` builds an nnls optimizer that has a higher number of max. iterations. This is optional!
- `nnls.py` contains the wrapper for the Fortran code of the adapted `nnls` method. See the optional part below. 
- `plot_misc.py` generates Figures 1, 2, 3, and 9.
- `plot_results.py` generates all the figures depicting the results using the `.npz` files created by `run_experiment.py`.
- `plot_timing.py` generates Figure 6.
- `plot_variables.py` contains some variables, e.g., data set sizes.
- `run_experiment.py` is used to run experiments. It will create result files of the following form `result_DATASET_K_INIT_M_PREPROCESSING_REPS_MAXITER_M.npz`
- `table.py` creates Figures 5 and 16.
- `table_worst.py` creates Figures 8 and 17.
- `utils.py` contains the implementation of the data loading and pre-processing functions.

### Optional

There is an implementation of the non-negative least squares method [`scipy.optimize.nnls`](https://docs.scipy.org/doc/scipy-1.9.0/reference/generated/scipy.optimize.nnls.html) which can be used. However, for some rare cases it works better when the number of maximum iterations is higher. We adapted the Fortran code used in `scipy` and increased the number of maximum iterations. It can be built by running
``` bash
bash build.sh
```
Note that this is *optional* and if you choose not to do it, the code will fall back to the implementation in [`scipy.optimize.nnls`](https://docs.scipy.org/doc/scipy-1.9.0/reference/generated/scipy.optimize.nnls.html).

## Data

We use the following 15 data sets:

| Data Set Name | Number of Data Points | Number of Dimensions |
|--------------:|----------------------:|---------------------:|
| [**California Housing**](https://scikit-learn.org/1.1/modules/generated/sklearn.datasets.fetch_california_housing.html) | 20,640 | 8 |
| [**Covertype**](https://archive.ics.uci.edu/ml/datasets/covertype) | 581,012 | 54 |
| [**KDD-Protein**](http://osmot.cs.cornell.edu/kddcup/datasets.html) | 145,751 | 74 |
| [**Pose**](http://vision.imar.ro/human3.6m/challenge_open.php) | 35,832 | 48 |
| [**RNA**](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) | 488,565 | 8 |
| [**Song**](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd) | 515,345 | 90 |
| [**FMA**](https://github.com/mdeff/fma) | 106,574 | 518 | 
| [**Airfoil**](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) | 1,503 | 5 |
| [**Concrete**](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) | 1,030 | 8 |
| [**Banking1**](https://www.people.vcu.edu/~jdula/FramesAlgorithms/BankingData/) | 4,971 | 7 |
| [**Banking2**](https://www.people.vcu.edu/~jdula/FramesAlgorithms/BankingData/) | 12,456 | 8 |
| [**Banking3**](https://www.people.vcu.edu/~jdula/FramesAlgorithms/BankingData/) | 19,939 | 11 |
| [**Ijcnn1**](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) | 49,990 | 22 |
| [**MiniBooNE**](https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification) | 130,064 | 50 |
| [**SUN Attribute**](https://cs.brown.edu/~gmpatter/sunattributes.html) | 14,340 | 102 |

The sources are stated in the paper and in `utils.py`.

## Running Experiments

To run all experiments, i.e., for the `Ijcnn1` data set, one would have to execute:
``` bash
for pre in CenterAndMaxScale Standardize; do
  for dataset in ijcnn1; do
    for k in 15 25 50 75 100; do
      for init in Uniform FurthestFirst FurthestSum KMpp AApp AAcoreset; do
        python run_experiment.py --dataset=$dataset --k=$k --preprocessing=$pre --init_method=$init --m=0 --repetitions=30;
      done;
      for m in 1 5 10 20; do
        python run_experiment.py --dataset=$dataset --k=$k --preprocessing=$pre --init_method=AAppMC --m=$m --repetitions=30;
      done;
    done;
  done;
done
```
This can be split to multiple nodes. Every run of `run_experiment.py` will create a result file of the following form: `result_DATASET_K_INIT_M_PREPROCESSING_REPS_MAXITER_M.npz`. The number of maximum iterations is set to 30.


