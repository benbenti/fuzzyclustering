# Fuzzy clustering

This package implements the fuzzy c-means (FCM) classification algorithm, as well as a set of graphic tools to visualise the classification outcomes.

The FCM performs a soft classification. Instead of being assigned to a single category, each sample is given a membership score (akin to a probability of belonging) to every category. The algorithm iteratively uses the membership scores to update the position of the cluster centroids, and the position of the cluster centroids to update the membership scores. The classical FCM [1] is known to be sensitive to high dimensionality [2]. I'm working on the implementation of two modifications of the algorithm to improve classification outcomes: the polynomial fuzzifier function and membership regularisation [3].

This package aims to propose three methods to assess the quality of the final classification results. So far, only the first one if fully functional:

- *the objective function of the FCM algorithm*. It takes mostly the compactness of the clusters into account.
- *the VIdso index* [4]. It combines measures of cluster dispersion, separation, and overlap (**work in progress**).
- *the generalised intra-inter silhouette* [5]. It combines cluster compactness and separation, and provides a sample-by-sample measure of assignment quality. However, this index comes with high computational costs (**work in progress**).

The visualisation tools include a graphical aid to identify clustering solutions which probably highlight an underlying structure in the dataset, a quantitative measure and display of whether and how much samples are representative of a cluster (the typicality), and triangular plots to visualise the gradation of samples between cluster centroids. Additionally, if another partition of the dataset is available, there is a partition comparison display.

Finally, I implemented two methods for the 2D visualisation of high-dimensional datasets: a first one based on Principal Component Analysis (PCA) which relies on linear projections, and a second one based on the t-SNE algorithm [6] which performs a non-linear transformation of the dataset.

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

**The fuzzifier of the classical FCM range from 1 to +inf**. A fuzzifier of 1 corresponds to a hard classification. The larger the fuzzifier, the more clusters are allowed to overlap.

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

### The FCM with polynomial fuzzifier [3]

**(work in progress)**

### The FCM with membership regularisation [3]

#### Using Shannon's entropy

 **(work in progress)**

#### Using quadratic entropy

 **(work in progress)**

## How to install/uninstall the package

 **(work in progress)**

## Tutorial

 **(work in progress)**

## Bibliography

- [1] **Bezdek JC, Ehrlich R, Full W** (1981) FCM: the fuzzy c-means algorithm. *Computer & Geosciences 10(2-3):191-203*. DOI:10.1016/0098-3004(84)90020-7
- [2] **Winkler R, Klawonn R, Kruse R** (2010) Fuzzy c-means in high-dimensional spaces. *International Journal of Fuzzy System Applications*. DOI:10.4018/ijfsa.2011010101.
- [3] **Borgelt C** (2013) Objective functions for fuzzy clustering. In *Computational intelligence and intelligent data analysis, pp 3-16*. DOI:10.1007/978-3-642-32378-2_1
- [4] **Bharill N, Tiwari A** (2014) Enhanced cluster validity index for the evaluation of optimal number of clusters for fuzzy c-means algorithm. *IEEE international conference on fuzzy systems*. DOI:10.1109/FUZZ-IEEE.2014.6891591
- [5] **Rawashdeh M, Ralescu A** (2012) Crisp and fuzzy cluster validity: generalised intra-inter silhouette. *Annual meeting of the north American fuzzy information processing society*. DOI:10.1109/NAFIPS.2012.6290969
- [6] **van der Maaten L, Hinton G** (2008) Visualizing Data using t-SNE. *Journal of Machine Learning Research*.
