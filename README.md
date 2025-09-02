# Fuzzy clustering

This package implements the fuzzy c-means (FCM) classification algorithm, as well as a set of graphic tools to visualise the classification outcomes.

The FCM performs a soft classification. Instead of being assigned to a single category, each sample is given a membership score (akin to a probability of belonging) to every category. The algorithm iteratively uses the membership scores to update the position of the cluster centroids, and the position of the cluster centroids to update the membership scores. The classical FCM [1] is known to be sensitive to high dimensionality [2]. I'm working on the implementation of two modifications of the algorithm to improve classification outcomes: the polynomial fuzzifier function and membership regularisation [3].

This package aims to propose three methods to assess the quality of the final classification results. So far, only the first one if fully functional:

- *the objective function of the FCM algorithm*. It takes mostly the compactness of the clusters into account.
- *the VIdso index* [4]. It combines measures of cluster dispersion, separation, and overlap (**work in progress**).
- *the generalised intra-inter silhouette* [5]. It combines cluster compactness and separation, and provides a sample-by-sample measure of assignment quality. However, this index comes with high computational costs (**work in progress**).

The visualisation tools include a graphical aid to identify clustering solutions which probably highlight an underlying structure in the dataset, a quantitative measure and display of whether and how much samples are representative of a cluster (the typicality), and triangular plots to visualise the gradation of samples between cluster centroids. Additionally, if another partition of the dataset is available, there is a partition comparison display.

Finally, I implemented three methods for the 2D visualisation of high-dimensional datasets, based on PCA and UMAP.

## Licence

This software is under a MIT License (see LICENSE.txt).

## The FCM algorithm

Let *X* be the dataset we want to classify. Let *n_s_* be the number of samples in the dataset. Each sample is described by *n_f_* features.  *X* is the set of samples *x_i_* with *i in [1, n_s_]*, each defined by features *x_i,j_* with *j in [1, n_f_]*. The fuzzy c-means algorithm aims to partition this dataset in a soft, fuzzy way. The flexibility of the classification is driven by *p*, the fuzzifier parameter. The expression of the fuzzifier differs across algorithm versions, but it always quantifies how crisp or fuzzy the partition can be. The fuzzy partition is defined by two matrices:

1. a matrix of cluster centroids *C*. Let *n_c_*  be the number of clusters. For each *k* in *[1, n_c_]*, the cluster centroid *c_k_* is defined by its position in the feature space.

2. a matrix of membership score *M*. Each sample *x_i_, i in [1, n_s_]* is assigned a membership score *m_i,k_* to cluster *k, k in [1, n_c_]*. Membership scores range between 0 and 1: *m_i,k_* corresponds to the probability that sample *i* belongs to cluster *k*. For each sample *i* and each cluster *k*:

$$m_{i,k} \in [0, 1]$$

$$\sum_{i=1}^{n_s} m_{i,k} = 1$$

The cluster centroids are initiated randomly from a uniform distribution over the range of features in the dataset. Then, the algorithms iteratively calculate memberships scores from the positions of data points and cluster centroids, and update the positions of the cluster centroids based on the positions and membership scores of data points. This iterative process continues until the convergence of an objective function *F(X, M, C)* or until a maximal number of iterations is reached.

The different versions of the FCM use specific formulas to calculate membership scores, to update the positions of cluster centroids, and to evaluate the objective function.

### The classical FCM [1]

**The fuzzifier of the classical FCM range from 1 to +inf (optimal range 1.5-2.5)**. A fuzzifier of 1 corresponds to a hard classification. The larger the fuzzifier, the more clusters are allowed to overlap.

The objective function of the classical FCM is:

$$F(X,C,M)=\sum_{i=1}^{n_s}\sum_{k=1}^{n_c}m_{ik}^p \times d_{ik}^2$$

With:

- *F(X, C, M)* the value of the objective function;
- *m_i,k_* the membership score of sample *i* to cluster *k*;
- *p* the fuzzifier;
- *d_i,k_* the Euclidian distance between sample *i*  and cluster *k* in the feature space.

The membership scores are calculated as:

$$\frac{1}{m_{i,a}}=\sum_{k=1}^{n_c} \biggl(\frac{d_{i,a}}{d_{i,c}}\biggr)^\frac{2}{p-1}$$

With:

- *m_i,a_* the membership score of sample *i* to cluster *a*;
- *d_i,a_* the euclidian distance between sample *i* and cluster *a*;
- *p* the fuzzifier.

The position of the cluster centroids are updated using the following formula:

$$c_k=\frac{\sum_{i=1}^{n_s}m_{i,k}^p \times x_{i}}{\sum_{i=1}^{n_s}m_{i,k}^p}$$

With:

- *c_k_* the position of the centroid of cluster *k* in the feature space;
- *m_i,k_* the membership score of sample *i* to cluster *k*;
- *p* the fuzzifier;
- *x_i_* the position of sample *i* in the feature space.

### Alternative FC algorithms he FCM with polynomial fuzzifier [3]

Several modifications of the FC algorithm exist - using a polynomial fuzzifier, or regulating membership scores with an entropy term (either using Shannon's or quadratic entropy) [3].
These alternative algorithms are not yet implemented in the package.

## How to install/uninstall the package

(no release yet, please make a local clone)

## Tutorial

### Extraction of acoustic features

See the documentation of the pylotwhale package for details. Example below.

```
import pandas as pd
import numpy as np

import pylotwhale.signalProcessing.signalTools as sT
import pylotwhale.utils.dataTools as daT
import pylotwhale.utils.whaleFileProcessing as wP
import pylotwhale.MLwhales.featureExtraction as fex
import pylotwhale.MLwhales.MLtools_beta as myML

# Load audio data and annotations
df = pd.read_csv(callColl, usecols = ['file', 'label'])
wavColl = np.array(df, dtype=object)

# Build feature extraction pipeline
## Preprocessing
fs = 48000
T_settings = []
filt = 'band_pass_filter'
filtDi = {'fs': fs, 'lowcut':1000, 'highcut': 22000, 'order': 4}
T_settings.append(('bandFilter', (filt, filtDi)))
prepro = 'maxabs_scale'
preproDict = {}
T_settings.append(('normaliseWF', (prepro, preproDict)))
## Define audio features
audioF = 'MFCC'
auD = {}
auD['fs'] = fs
auD['NFFT'] = 1024
auD['overlap'] = 0.5
auD['n_mels'] = 40
auD['Nceps'] = 40
T_settings.append(('Audio_features', (audioF, auD)))
## Summarisation method.
summDict = {'Nslices': nslices, 'normalise': True}
summType = 'splitting'
T_settings.append(('summ', (summType, summDict)))
## Make transformation pipeline
Tpipe = fex.makeTransformationsPipeline(T_settings)
feExFun = Tpipe.fun
    
# Extract raw features from wav files
datO = myML.dataXy_names()
datO_new = fex.wavLCollection2datXy(wavColl, featExtFun=feExFun, fs=fs)
datO.addInstances(datO_new.X, datO_new.y_names)
```

### Run fuzzy clustering

```
import lib.algorithms as al

dataset = datO.X
fuzz_interval = [1, 2.5]  # Interval of fuzziness over which to run the algorithm.
step = 0.01  # Fuzziness steps
kmax = 15  # Maximum number of fuzzy clusters.
algo = al.fuzzyClustering  # which algorithm to use (alternative algorithms in construction)
maxiter = 1000  # Maximal number of iterations for the FC algorithm
err = 0.0001  # Minimal improvement of the objective function under which to stop iterating

FC = al.full_process(dataset, fuzz_interval, step, kmax, algo, maxiter, err)

# Possibility to run in parallel with larger steps (for instance five terminals
# with a step of 0.05 and p_max differing by 0.01)
# Requires to merge the partial results dictionnaries afterwards
# full_dict = {**dict1, ..., **dictn}
```

### Select interesting clustering solution

The first step of the analysis is to select interesting clustering solutions.
```
import al.visuals as vis

# Identify clustering solutions of interest.
tab = vis.identify_stable_solutions(FC, plot=True)
```

`tab` is a np.array containing the optimal fuzzy clustering outcomes for all values of fuzziness. This function can plot the optimal number of clusters relative to fuzziness values. Clustering solutions that are stable over a large range of fuzziness may represent underlying structures in the dataset and warrant further investigation.

### Visualise the outcome of a single fuzzy clustering run.

The `visuals` module provides tools for the 2D-visualisation of fuzzy clustering results. For both function, samples can be coloured according to the fuzzy cluster they are most similar to, or according to user-provided categories.

```
# fc is an instance of the FuzzyClustering class storing the results of fuzzy clustering

fig, ax = PCA_plot(fc)  # using Principal Component Analysis
fig, ax = UMAP_plot(fc)  # using Uniform Manifold Approximation and Projection
```

Then, it is possible to plot histograms of typicality. Typicality is measured as the difference between the two highest membership scores of a sample and is indicates whether samples are similar to cluster centres (stereotypes samples, typicality close to 1) or not (graded samples, typicality close to 0).

```
# fc is an instance of the FuzzyClustering class.
fig, ax = plot_typicality(fc)
```

Triangular gradation plots are a second, more detailed visualisation of call gradation across fuzzy clusters. They plot the membership scores of samples to two different fuzzy clusters. Stereotyped samples are location around the tips of the triangular plot, whereas graded samples are located in the diagonal and the center of the plot.

```
# fc is an instance of the FuzzyClustering class.
# c1 and c2 are two integers pointing to fuzzy cluster indices.

# Plot a single gradation plot for clusters c1 and c2.
fig, ax = triangular_gradation_plot(fc, c1, c2)

# Plot gradation plots for all pairs of fuzzy clusters.
fig, ax = triangular_gradation_plots(fc)
```

Finally, the module provides tools to compare the fuzzy clustering results with another, user-provided classification.

```
# fc is an instance of the FuzzyClustering class.
# partition is a user-provided classfication of the samples (np.array, size=(n_samples, 1))
# name_list is the list of the categories in the user-provided classification.

c_mat = make_partition_comparison(fc, partition)  # confusion matrix crossing both classficiation.
fig, ax = plot_partition_comparison(c_mat, name_list)
```

Please refer to our study of long-finned pilot whale calls [6] for an example of how to use the visualisation tools to analyse fuzzy clustering results.

## Bibliography

- [1] **Bezdek JC, Ehrlich R, Full W** (1981) FCM: the fuzzy c-means algorithm. *Computer & Geosciences 10(2-3):191-203*. DOI:10.1016/0098-3004(84)90020-7
- [2] **Winkler R, Klawonn R, Kruse R** (2010) Fuzzy c-means in high-dimensional spaces. *International Journal of Fuzzy System Applications*. DOI:10.4018/ijfsa.2011010101.
- [3] **Borgelt C** (2013) Objective functions for fuzzy clustering. In *Computational intelligence and intelligent data analysis, pp 3-16*. DOI:10.1007/978-3-642-32378-2_1
- [4] **Bharill N, Tiwari A** (2014) Enhanced cluster validity index for the evaluation of optimal number of clusters for fuzzy c-means algorithm. *IEEE international conference on fuzzy systems*. DOI:10.1109/FUZZ-IEEE.2014.6891591
- [5] **Rawashdeh M, Ralescu A** (2012) Crisp and fuzzy cluster validity: generalised intra-inter silhouette. *Annual meeting of the north American fuzzy information processing society*. DOI:10.1109/NAFIPS.2012.6290969
- [6] **Benti B, Miller PJO, Vester H, Noriega F, and Curé C** (2024) Unsupervised classification of graded animal sounds using fuzzy clustering. *bioArkiv* 2024.09.13.612808. DOI:10.1101/2024.09.13.612808
