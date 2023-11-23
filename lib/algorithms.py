"""
This module implements the basic version of the fuzzy c-means (FCM)
classification algorithm.
-----------------------------------------------------------------------
The FCM defines a number of clusters, and gives each sample a
membership score to each cluster (akin to a probability of belonging).
Clusters are initiated randomly, then the algorithm iteratively uses
membership scores to update the position of cluster centroids, and the
position of cluster centroids to update membership scores.

This module propose three methods to evaluate clustering quality:
- the objective function of the FCM (cluster compactness)
- the VIdso index (cluster dispersion, separation, and overlap)
- the generalised intra-inter silhouette (cluster compactness and
    separation, computationally expensive).
-----------------------------------------------------------------------
CLASSES
-------
FuzzyClustering
-----------------------------------------------------------------------
FUNCTIONS
---------
classification
compare_quality
full_process
euclidian_distance
"""

import numpy as np
import random
import math
import itertools


class FuzzyClustering():
    """
    FuzzyClustering(dataset, p, nc)

    A FuzzyClustering object summarises a fuzzy c-means classification.
    It stores the dataset, performs the basic steps of the algorithm,
    and stores the resulting partition (as the position of cluster
    centroids and membership scores).
    FuzzyClustering objects retain the history of the cluster centroid
    positions and the values of the objective function to document the
    algorithm progress.

    Parameters
    ----------
    dataset (2d array)
        The dataset to classify. Samples in row, features in columns.
    p (float)
        Value of the fuzzifier parameter.
    nc (integer)
        Number of clusters.

    Notes
    -----
    The fuzzifier parameter should be larger than 1.

    Attributes
    ----------
    data (2d array)
        The dataset to classify. Samples in rows, features in columns.
    n_samples (integer)
        Number of samples in the dataset.
    fuzzifier (float)
        Value of the fuzzifier parameter.
    clusters (list of 2d arrays)
        Successive positions of the cluster centroids in the feature
        space. For each element of the list, cluster centroids in rows,
        features in columns.
    n_clusters (integer)
        Number of clusters
    memberships (2d array)
        Membership scores. Samples in rows, clusters in columns.
    obj_function (list of floats)
        Successive values of the objective function.
    cluster_quality (float or list of floats)
        Quality index for the final clustering solution. Single value
        if evaluated as the objective function, 3-element list if
        evaluated as the VIdso index, n_samples-element list if
        evaluated as the generalised intra-inter silhouette.

    Methods
    -------
    __init__
        Creates a new FuzzyClustering instance.
    _f
        Membership score modification function (fuzzification function)
        used to evaluate the objective function and to update the
        position of cluster centroids.
    _g
        Entropy term used to evaluate the objective function.
    initiate_clusters
        Creates initial random clusters.
    calculate_memberships
        Calculates the membership scores from the distances between
        data points and cluster centroids, and fuzzifier value.
    update_clusters
        Updates the position of the cluster centroids from the
        membership scores and the position of data points in the
        feature space.
    evaluate_objective_function
        Evaluates the value of the objective function.
    fuse_centroids
        Fuses centroids which are close from each other.
    VIdso
        Computes the dispersion, separation, and overlap indices of
        clustering quality.
    intrainter_silhouette
        Computes the generalised intra-inter silhouette.
    """
    min_fuzzifier = 1
    max_fuzzifier = np.inf

    def __init__(self, dataset, p, nc):
        """
        Creates a new FuzzyClustering instance.

        Parameters
        ----------
        dataset (2d array)
            Dataset to classify. Samples in rows, features in columns.
        p (float)
            Value of the fuzzifier parameter.
        nc (integer)
            Number of clusters.
        """
        self.data = dataset
        self.n_samples = self.data.shape[0]
        self.fuzzifier = p
        self.n_clusters = nc
        self.clusters = []
        self.memberships = None
        self.obj_function = []
        self.cluster_quality = None
        # clusters and obj_function are initiated as empty lists to
        # enable appending during the classification process.

    def _f(self):
        """
        Membership score modification function (fuzzification function)
        used to evaluate the objective function and update the position
        of cluster centroids.
        """
        return np.power(self.memberships, self.fuzzifier)

    def _g(self):
        """
        Entropy term used to evaluate the objective function.
        """
        return 0

    def initiate_clusters(self, s=None):
        """
        Creates self.n_clusters random cluster centroids.

        Parameters
        ----------
        s (integer, string, bytes, bytearray)
            The seed of the random number generator (see random).
            None seeds from current time or based on operating system.

        Notes
        -----
        The cluster centroids are drawn from a uniform distribution
        over the range of each feature in the dataset.
        """
        rng = random.Random()
        rng.seed(s)
        mins = np.min(self.data, axis=0)
        maxs = np.max(self.data, axis=0)
        self.clusters.append(np.array([[rng.uniform(a, b)
                                        for a, b in zip(mins, maxs)
                                        ]
                                       for i in range(self.n_clusters)
                                       ]
                                      )
                             )

    def calculate_memberships(self):
        """
        Calculates membership scores from the distances between samples
        and cluster centroids, and fuzzifier value.
        """
        ct = self.clusters[-1]
        d = np.array([[euclidian_distance(i, c)
                       for c in ct
                       ]
                      for i in self.data
                      ]
                     )
        common_term = np.sum(d ** (2/(1-self.fuzzifier-1)), axis=1)
        tmp = np.array([common_term[i] * d[i] ** (2/(self.fuzzifier-1))
                        for i in range(self.n_samples)
                        ]
                       )
        mb = tmp ** (-1)
        # Membership score may be nan if the distance between a datapoint and
        # a cluster centroid is null.
        # Since the cluster centroid and the data point are at the same place,
        # set the membership score to 1.
        mb[np.isnan(mb)] = 1
        # Correct the scores so that the memberships of each sample sums up to
        # 1. This prevents the accumulation of floating point operation
        # errors.
        for i in range(self.n_samples):
            mb[i] /= sum(mb[i])
        self.memberships = mb

    def update_clusters(self):
        """
        Updates the position of the cluster centroids based on the
        position of data points and their fuzzified membership scores.
        """
        f_mb = self._f()  # Fuzzified membership scores.
        new_ct = np.zeros(shape=self.clusters[-1].shape)
        denoms = np.sum(f_mb, axis=0)
        for k in range(self.n_clusters):
            num = np.sum(np.array([f_mb[i, k] * self.data[i]
                                   for i in range(self.n_samples)
                                   ]
                                  ),
                         axis=0
                         )
            new_ct[k] = num / denoms[k]
        self.clusters.append(new_ct)

    def evaluate_objective_function(self):
        """
        Evaluates the value of the objective function.
        """
        ct = self.clusters[-1]
        f_mb = self._f()  # Fuzzified membership scores.
        g_mb = self._g()  # Entropy term.
        squared_dist = np.array([[euclidian_distance(i, c) ** 2
                                  for c in ct
                                  ]
                                 for i in self.data
                                 ]
                                )
        tmp1 = np.sum(np.multiply(f_mb, squared_dist))
        tmp2 = np.sum(g_mb)
        self.obj_function.append(tmp1 + tmp2)

    def fuse_centroids(self, d_th=None):
        """
        Fuses cluster centroids closer than the fusion threshold.
        By default, the fusion threshold corresponds to 0.01 % of the range of
        every feature in the dataset.
        """

        if d_th is None:
            d_th = 0.0001 * np.sum(np.abs(np.max(self.data, axis=0) -
                                          np.min(self.data, axis=0)
                                          )
                                   )

        cluster_dist = np.array([[euclidian_distance(c1, c2)
                                  for c1 in self.clusters[-1]
                                  ]
                                 for c2 in self.clusters[-1]
                                 ]
                                )
        close_cluster = (cluster_dist <= d_th)
        # Make lists of clusters close from each cluster.
        close_set = [list(set([j for j in range(self.n_clusters)
                               if close_cluster[i, j]
                               ]
                              )
                          )
                     for i in range(self.n_clusters)
                     ]
        if max([sum(i) for i in close_set]) == 1:  # No centroid fusion.
            return False
        else:
            # Fuse list of cluster clusters with common elements.
            for i, elt in enumerate(close_set):
                for j, lst in enumerate(close_set):
                    if any(k in elt for k in lst):
                        elt += [k for k in lst if k not in elt]
                elt.sort()
            # Keep unique sets.
            close_set_unique = []
            for elt in close_set:
                if elt not in close_set_unique:
                    close_set_unique.append(elt)
            # Compute new cluster (mean of close clusters)
            new_ct = np.empty(shape=(len(close_set_unique),
                                     self.data.shape[1]
                                     )
                              )
            for i, elt in enumerate(close_set_unique):
                new_ct[i] = np.mean(self.clusters[-1][elt, :], axis=0)
            # Append new cluster table in FuzzyClustering instance
            # and update cluster number.
            self.clusters.append(new_ct)
            self.n_clusters = new_ct.shape[0]
        return True

    def VIdso(self):
        """
        Evaluates the cluster dispersion, separation, and overlap
        indices of a given clustering solution.
        """
        ct = self.clusters[-1]
        mb = self.memberships
        # Dispersion index. Ratio of the dispersion around cluster centroids
        # over the dispersion around the center of gravity of the dataset.
        # Cluster centroids with many points concentrated around them have
        # lower values than cluster centroids with few points around.
        cv_data = np.std(self.data, axis=0) / abs(np.mean(self.data, axis=0))
        cv_c = np.zeros(shape=(self.n_clusters,))
        for k in range(self.n_clusters):
            sigma_c = np.sqrt(1/self.n_samples
                              * np.sum((self.data - ct[k]) ** 2,
                                       axis=0
                                       )
                              )
            cv_c[k] = np.max(np.multiply(sigma_c, abs(ct[k])**(-1)))
        disp = np.max(cv_c) / np.max(cv_data)
        # Separation and overlap indices. Minimal distance between clusters,
        # measured as the highest non-dominant membership scores between each
        # pair of clusters (separation).
        S = np.zeros(shape=(self.n_clusters, self.n_clusters))
        Ov = np.zeros(shape=S.shape)
        for c1, c2 in itertools.combinations(range(self.n_clusters), 2):
            c_max = np.max(mb[:, [c1, c2]], axis=1)  # Dominant membership.
            c_min = np.min(mb[:, [c1, c2]], axis=1)  # Dominee membership.
            # Similarity between clusters = highest non-dominant score.
            S[c1, c2] = np.max(c_min)
            # Overlap quantification for a data point. Decreases with
            # dominant membership score.
            R = -2 * c_max + 2
            # No overlap if complete membership or null membership.
            idx_mb = [i for i in range(self.n_samples)
                      if (c_max[i] == 1 or c_min[i] == 0)
                      ]
            R[idx_mb] = 0
            # Overlap cannot be higher than 1.
            idx_Rhigh = [i for i in range(self.n_samples) if R[i] > 1]
            R[idx_Rhigh] = 1
            Ov[c1, c2] = np.sum(R)
        sep = np.min(1 - S)
        ovlp = np.max(Ov)
        self.cluster_quality = [disp, sep, ovlp]

    def intrainter_silhouette(self):
        """
        Computes the generalised intra-inter silhouette of a given
        fuzzy partition.
        """
        mb = self.memberships
        a = np.zeros(shape=(self.n_samples,))
        b = np.zeros(shape=(self.n_samples,))
        # Measure the silhouette of each sample.
        for i in range(self.n_samples):
            dist = [euclidian_distance(self.data[i], self.data[j])
                    for j in range(self.n_samples)
                    ]
            # Intra silhouette.
            a[i] = np.inf
            for c in range(self.n_clusters):
                intra_c = [min([mb[i, c], mb[j, c]])
                           for j in range(self.n_samples)
                           ]
                a_tmp = np.sum(np.multiply(intra_c, dist)) / np.sum(intra_c)
                if a_tmp <= a[i]:
                    a[i] = a_tmp
            # Inter silhouette.
            b[i] = np.inf
            for c1, c2 in itertools.combinations(range(self.n_clusters), 2):
                inter_c1_c2 = [max([min([mb[i, c1], mb[j, c2]]),
                                    min([mb[i, c2], mb[j, c1]])
                                    ]
                                   )
                               for j in range(self.n_samples)
                               ]
                b_tmp = (np.sum(np.multiply(inter_c1_c2, dist))
                         / np.sum(inter_c1_c2)
                         )
                if b_tmp <= b[i]:
                    b[i] = b_tmp
        s = (b - a) / np.max(np.vstack((b, a)), axis=0)
        self.cluster_quality = s


def classification(dataset, p, nc, algo,
                   itermax, err, err_bis=None, seed=None, verbose=0):
    """
    Runs the fuzzy c-means algorithm for a pair of fuzzifier value and
    number of clusters. Makes 100 realisations of the algorithms with
    random initialisations.

    Parameters
    ----------
    dataset (2d array)
        The dataset to classify. Samples in rows, features in columns.
    p (float)
        Value of the fuzzifier. Should be within the interval of valid
        values depending on the FCM version.
    nc (integer)
        Number of clusters.
    algo (class)
        The version of the fuzzy c-means algorithm to use.
        FuzzyClustering, FuzzyClusteringPoly, FuzzyClusteringRegulSh,
        or FuzzyClusteringRegulQuad.
    itermax (integer)
        Maximum number of iterations to reach convergence.
    err (float)
        Minimum improvement of the objective function to continue the
        iterations.
    err_bis (float)
        Minimum difference in consecutive cluster centroid positions to
        continue the iteration.
        Default is None. If given, overrides err
    seed (integer, float, byte, or bytearray)
        Seed of the random number generator used to initiate random
        clusters.
    Returns
    -------
    FC_list (list)
        A 100-element list containing the final states of the FCM
        realisations.
    """
    if p < algo.min_fuzzifier or p > algo.max_fuzzifier:
        raise ValueError('Fuzzifier out of range')
    if err_bis:  # If stop iteration decided from cluster displacement.
        err = err_bis
    FC_list = [None] * 100
    for n in range(100):
        FC = algo(dataset, p, nc)
        FC.initiate_clusters(seed)
        FC.calculate_memberships()
        FC.evaluate_objective_function()
        stopiter = err + 1
        n_loops = 0
        while stopiter >= err and n_loops <= itermax:
            FC.update_clusters()
            fused = FC.fuse_centroids()
            if fused and verbose == 2:
                print('Cluster fusion!')
            FC.calculate_memberships()
            FC.evaluate_objective_function()
            if verbose == 2:
                print('Objective function = {}'.format(FC.obj_function[-1]))
            if fused:  # If fusion skip stopiter check and loop count.
                continue
            if not err_bis:  # Assess diminution of objective function.
                stopiter = FC.obj_function[-2] - FC.obj_function[-1]
            else:  # Assess displacement of fuzzy clusters.
                stopiter = max([euclidian_distance(FC.clusters[-1][k],
                                                   FC.clusters[-2][k]
                                                   )
                                for k in range(FC.n_clusters)
                                ]
                               )
            n_loops += 1
        FC_list[n] = FC
    return FC_list


def compare_quality(lst, q_method=None):
    """
    Compare the clustering quality of several fuzzy clustering
    solutions.

    Parameters
    ----------
    lst (list of FuzzyClustering instances)
        Clustering solutions to compare (must be partitions of the same
        dataset).
    q_method (FuzzyClustering class method)
        Method for the evaluation of clustering quality. FC.VIdso() or
        FC.intrainter_silhouette().
        Default is None and use the value of the objective function.

    Returns
    -------
    quality (list)
        List of clustering quality values for each fuzzy partition.
    sorted_lst (list)
        The positions of the elements of lst in decreasing clustering
        quality order.
    """
    if lst[0].cluster_quality is None:  # No computed quality yet.
        if q_method is None:  # Uses the objective funtion.
            for elt in lst:
                elt.cluster_quality = elt.obj_function[-1]
        else:  # Runs q_method.
            for elt in lst:
                q_method(elt)

    if isinstance(lst[0].cluster_quality, np.float):  # Objective function.
        quality = [elt.cluster_quality for elt in lst]
    elif len(lst[0].cluster_quality) == 3:  # Uses the VIdso index.
        disp = [elt.cluster_quality[0] for elt in lst]
        sep = [elt.cluster_quality[1] for elt in lst]
        ovlp = [elt.cluster_quality[2] for elt in lst]
        quality = [(d / max(disp) + o / max(ovlp)) / (s / max(sep))
                   for d, o, s in zip(disp, ovlp, sep)
                   ]
    elif len(lst[0].cluster_quality == lst[0].n_samples):  # Silhouette.
        quality = [np.mean(elt.cluster_quality) for elt in lst]
    # Sort clustering realisations according to quality
    sorted_lst = sorted(range(len(quality)), key=quality.__getitem__)
    return quality, sorted_lst


def full_process(dataset, fuzz_range, step, nc_max, algo,
                 itermax, err, err_bis=None, seed=None, q_method=None,
                 verbose=0
                 ):
    """
    Runs the full classification procedure. Runs the fuzzy c-means
    algorithm input on the dataset for each fuzzifier value in
    fuzzif_range (using increments of size step) and for 2, ..., nc_max
    clusters.At each step, makes 100 realisations and keeps the
    solution with the best clustering quality (according to q_method).

    Parameters
    ----------
    dataset (2d array)
        The dataset to classify. Samples in rows, features in columns.
    fuzz_range (2-float list)
        Boundaries of the fuzzifier range to cover. Raises a ValueError
        if outside of [algo.min_fuzzifier, algo.max_fuzzifier].
    step (float)
        Size of the increments to cover the range of fuzzifier values.
    nc_max (integer)
        Maximum number of clusters involved.
    algo (class).
        The version of the fuzzy c-means algorithm to use.
        FuzzyClustering, FuzzyClusteringPoly, FuzzyClusteringRegulSh,
        or FuzzyClusteringRegulQuad.
    itermax (integer)
        Maximum number of iterations to reach convergence.
    err (float)
        Minimum improvement of objective function to continue the
        iterations.
    err_bis (float)
        Minimum difference in cluster centroid position to continue
        the iteration.
        Default is None. If given, overrides err.
    seed (integer, float, byte, or bytearray)
        Seed of the random number generator used to initiate random
        clusters.
    q_method (class method):
        Method for the evaluation of clustering quality. FC.VIdso() or
        FC.intrainter_silhouette().
        Default is None and use the value of the objective function.
    verbose (int):
        Amount of verbal information to display. 0 displays no info.
        1 (default) prints a message when a (fuzzifier, n_cluster) loop is
        done. 2 additionally prints successive objective function values.

    Returns
    -------
    clustering_solution (dict)
        A dictionary of dictionaries storing the clustering solutions.
        Primary keys are the values of the fuzzifier parameter.
        Secondary keys are the numbers of clusters.
        The items of the secondary dictionaries are the FuzzyClustering
        instances corresponding to the fuzzifier value and number of
        clusters given as primary and secondary key.
    """
    cd1 = fuzz_range[0] < algo.min_fuzzifier
    cd2 = fuzz_range[1] > algo.max_fuzzifier
    if cd1 or cd2:
        raise ValueError('Fuzzifier out of range')
    clustering_solution = {}
    if err_bis:
        err = err_bis
    for p in np.arange(fuzz_range[1], fuzz_range[0], -step):
        clustering_solution[round(p, 2)] = {}
        for k in range(2, nc_max+1):
            # Make 100 runs of the algorithm.
            results = classification(dataset, p, k, algo,
                                     itermax, err, err_bis, seed, verbose
                                     )
            # Compare quality and keep best clustering solution.
            q, idx = compare_quality(results, q_method)
            clustering_solution[round(p, 2)][k] = results[idx[0]]
            if verbose:
                print('Fuzzifier {} and {} clusters: done'.format(round(p, 2),
                                                                  k
                                                                  )
                      )
    return clustering_solution


def euclidian_distance(A, B):
    """
    Measures the euclidian distance between two n-dimension points.
    """
    return math.sqrt(np.sum((A - B) ** 2))
