# Fuzzy clustering
This package implements four different versions of the fuzzy c-means (FCM) classification algorithm, as well as a set of graphic tools to visualise the classification outcomes.

The FCM performs a soft classification. Instead of being assigned to a single category, each sample is given a membership score (akin to a probability of belonging) to every category. The algorithm iteratively uses the membership scores to update the position of the cluster centroids, and the position of the cluster centroids to update the membership scores. The classical FCM [1] is known to be sensitive to high dimensionality [2]. We implemented two modifications of the algorithm to improve classification outcomes: the polynomial fuzzifier function and membership regularisation [3]. We implemented membership regularisation using either Shannon or quadratic entropy.

This package proposes three methods to assess the quality of the final classification results: 1) <i>the objective function of the FCM algorithms</i>. It takes mostly the compactness of the clusters into account 2) <i>the VIdso index</i> [4]. It combines measures of cluster dispersion, separation, and overlap 3) <i>the generalised intra-inter silhouette</i> [5]. It combines cluster compactness and separation, and provides a sample-by-sample measure of assignment quality. However, this index comes with high computational costs.

We implemented tools to visualise the outcomes of the FCM classification. These tools include a graphical aid to identify clustering solutions which probably highlight an underlying structure in the dataset, a quantitative measure and display of whether and how much samples are representative of a cluster (the typicality), and triangular plots to visualise the gradation of samples between cluster centroids. Additionally, if another partition of the dataset is available, this module can compare both classification. Finally, we implemented two methods for the 2D visualisation of high-dimensional dataset: a first one based on Principal Component Analysis (PCA) which relies on linear projections, and a second one based on the t-SNE algorithms [6] which performs a non-linear transformation of the dataset.
## Licence
This software is under a MIT License (see LICENSE.txt).

## The four versions of the FCM algorithm

Let <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CX"> be the dataset we want to classify. Let <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CN_s"> be the number of samples in the dataset. Each sample is described by <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CN_f"> features. Therefore <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CX=\{x_{ij}\}, i\in[1, N_s], j\in[1, N_f]">. All versions of the fuzzy c-means algorithms aim to partition this dataset in a soft, fuzzy way. The flexibility of the classification is driven by <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cp">, the fuzzifier parameter. The expression of the fuzzifier differs across algorithm versions, but it always quantifies how crisp or fuzzy the partition can be. The fuzzy partition is defined by two matrices:
- a matrix of cluster centroids <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CC">. Let <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CN_c"> be the number of cluster. For each <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck\in[1, N_c]">, the cluster centroid <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k"> is defined by its position in the feature space: <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k=\{c_{kj}\}, j\in[1, N_f]">.
- a matrix of membership score <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CM">. Each sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci\in[1, N_s]"> is assigned a membership score to cluster
<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_1,...,c_k,...c_{N_c}">. Membership scores range between 0 and 1: <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> corresponds to the probability that sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> belongs to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">. For each sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci\in[1, N_s]">, <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Csum_{k=1}^{N_c} m_{ik}=1">.

The cluster centroids are initiated randomly from a uniform distribution over the range of features in the dataset. Then, the algorithms iteratively calculate memberships scores from the positions of data points and cluster centroids, and update the positions of the cluster centroids based on the positions and membership scores of data points. This iterative process continues until the convergence of an objective function <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CO(X, C, M)"> or until a fixed number of iterations is reached.

The four versions of the FCM differ in the exact formula they use to calculate membership scores, update the positions of cluster centroids, and evaluate the objective function.

### The classifical FCM [1]

__The fuzzifier of the classical FCM range from 1 to +inf__. A fuzzifier of 1 corresponds to a hard classification. The larger the fuzzifier, the more clusters are allowed to overlap.

The objective function of the classical FCM is:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CO(X,C,M)=\sum_{i=1}^{N_s}\sum_{k=1}^{N_c}m_{ik}^pd_{ik}^2">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CO(X,C,M)"> is the value of the objective function
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cp"> is the fuzzifier
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cd_{ik}"> is the euclidian distance between sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> and cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck"> in the feature space.

The membership scores are calculated as:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}=\frac{d_{ik}^{\frac{2}{1-p}}}{\sum_{c=1}^{N_c}d_{ic}^{\frac{2}{1-p}}}">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cd_{ik}"> is the euclidian distance between sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> and cluster centroid <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cp"> is the fuzzifier

The position of the cluster centroids are updated using the following formula:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k=\frac{\sum_{i=1}^{N_s}m_{ik}^px_{i}}{\sum_{i=1}^{N_s}m_{ik}^p}">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k"> is the position of cluster centroid <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cp"> is the fuzzifier

### The FCM with polynomial fuzzifier [3]

The polynomial fuzzifier introduces an area of crisp clustering around cluster centroids. __The fuzzifier ranges from 0 to 1__, and it corresponds to the ratio of the smaller over the larger distance below which clustering becomes crisp.

The objective function of the FCM with polynomial fuzzifier is:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CO(X,C,M)=\sum_{i=1}^{N_s}\sum_{k=1}^{N_c}(\frac{1-p}{1%2Bp}m_{ik}^2%2B\frac{2p}{1%2Bp}m_{ik})d_{ik}^2">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CO(X,C,M)"> is the value of the objective function
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cp"> is the fuzzifier
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cd_{ik}"> is the euclidian distance between sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> and cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck"> in the feature space.

The membership scores are calculated as:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}=\frac{m'_{ik}}{\sum_{c=1}^{N_c}m'_{ic}}">

With: <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm'_{ik}=max\{0, d_{ik}^{-2}-\frac{p}{(1%2Bp)(c_k-1)}\sum_{c=1}^{c_k}d_{is(c)}^{-2}\}">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cs(c)"> is the index of the clusters sorted by ascending distance to sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k=max\{c \mid d_{is(c)}^{-2}>\frac{p}{(1%2Bp)(c-1)}\sum_{l=1}^{s(c)}d_{is(l)}^{-2}\}">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cd_{ik}"> is the euclidian distance between sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> and cluster centroid <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cp"> is the fuzzifier

The position of the cluster centroids are updated using the following formula:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k=\frac{\sum_{i=1}^{N_s}(\frac{1-p}{1%2Bp}m_{ik}^2%2B\frac{2p}{1%2Bp}m_{ik})x_{i}}{\sum_{i=1}^{N_s}m_{ik}^p}">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k"> is the position of cluster centroid <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cp"> is the fuzzifier

### The FCM with membership regularisation [3]

Instead of changing how the fuzzifier intervenes in the calculation of the membership scores, membership regularisation adds an entropy term to the objective function, which draws the clustering solution away from crisp clustering.

#### Using Shannon's entropy

Using Shannon's entropy, __the fuzzifier ranges from 0 to +inf__. The update rule corresponds to calculating the probability that samples are drawn from Gaussian distributions centered at <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k"> with variance <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cfrac{1}{2}p">.

The objective function of the FCM with Shannon's entropy membership regularisation is:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CO(X,C,M)=\sum_{i=1}^{N_s}\sum_{k=1}^{N_c}m_{ik}d_{ik}^2%2Bp\sum_{i=1}^{N_s}\sum_{k=1}^{N_c}m_{ik}ln(m_{ik})">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CO(X,C,M)"> is the value of the objective function
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cp"> is the fuzzifier
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cd_{ik}"> is the euclidian distance between sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> and cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck"> in the feature space.

The membership scores are calculated as:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}=\frac{e^{\frac{-d_{ik}^2}{p}}}{\sum_{c=1}^{N_c}e^{\frac{-d_{ic}^2}{p}}}">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cd_{ik}"> is the euclidian distance between sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> and cluster centroid <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cp"> is the fuzzifier

The position of the cluster centroids are updated using the following formula:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k=\frac{\sum_{i=1}^{N_s}m_{ik}x_{i}}{\sum_{i=1}^{N_s}m_{ik}}">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k"> is the position of cluster centroid <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">

#### Using quadratic entropy

Quadratic entropy introduces areas of crisp clustering, similarly to the polynomial fuzzifier function. __The fuzzifier ranges from 0 to +inf__, and <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5C2p">
 can be seen as the difference between the distances between a sample and two clusters above which a crisp assignment is used.

The objective function of the FCM with quadratic entropy membership regularisation is:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CO(X,C,M)=\sum_{i=1}^{N_s}\sum_{k=1}^{N_c}m_{ik}d_{ik}^2%2Bp\sum_{i=1}^{N_s}\sum_{k=1}^{N_c}m_{ik}^2">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5CO(X,C,M)"> is the value of the objective function
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cp"> is the fuzzifier
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cd_{ik}"> is the euclidian distance between sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> and cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck"> in the feature space.

The membership scores are calculated as:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}=max\{0,\frac{1}{c_k}(1%2B\sum_{c=1}^{c_k}\frac{d_{is(c)}^2}{2p})-\frac{d_{ik}}{2p}\}">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cs(c)"> is the index of the clusters sorted by ascending distance to sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k=max\{c \mid \sum_{l=1}^{c}d_{is(l)}^2>cd_{ic}%2B2p\}">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cd_{ik}"> is the euclidian distance between sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> and cluster centroid <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cp"> is the fuzzifier

The position of the cluster centroids are updated using the following formula:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k=\frac{m_{ik}x_i}{\sum_{i=1}^{N_s}m_{ik}}">

Where:
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cc_k"> is the position of cluster centroid <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">
- <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cm_{ik}"> is the membership score of sample <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ci"> to cluster <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Ck">

## How to install/uninstall the package

incoming

## Tutorial (with the test dataset)

incoming

## Bibliography

[1] <b>Bezdek JC, Ehrlich R, Full W</b> (1981) FCM: the fuzzy c-means algorithm. <i>Computer & Geosciences</i> 10(2-3):191-203.
    DOI:10.1016/0098-3004(84)90020-7

[2] <b>Winkler R, Klawonn R, Kruse R</b> (2010) Fuzzy c-means in high-dimensional spaces. <i>International Journal of Fuzzy System Applications</i>
    DOI:10.4018/ijfsa.2011010101.

[3] <b>Borgelt C</b> (2013) Objective functions for fuzzy clustering. In <i>Computational intelligence and intelligent data analysis</i>, pp 3-16.
    DOI:10.1007/978-3-642-32378-2_1

[4] <b>Bharill N, Tiwari A</b> (2014) Enhanced cluster validity index for the evaluation of optimal number of clusters for fuzzy c-means algorithm. <i>IEEE international conference on fuzzy systems</i>.
    DOI:10.1109/FUZZ-IEEE.2014.6891591

[5] <b>Rawashdeh M, Ralescu A</b> (2012) Crisp and fuzzy cluster validity: generalised intra-inter silhouette. <i>Annual meeting of the north American fuzzy information processing society</i>.
    DOI:10.1109/NAFIPS.2012.6290969

[6] <b>van der Maaten L, Hinton G</b> (2008) Visualizing Data using t-SNE. <i>Journal of Machine Learning Research</i>.
