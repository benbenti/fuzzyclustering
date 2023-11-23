"""
This module implements several alternative versions of the FCM algo.
-----------------------------------------------------------------------
This module implements three versions of the FCM algorithm:
- the algorithm with a polynomial fuzzifier function  (in progress)
- the algorithm with Shannon entropy-based membership regularisation
(in progress).
- the algorithm with quadratic entropy-based membership regularisation
(in progress).
-----------------------------------------------------------------------
CLASSES
-------
FuzzyClusteringPoly (in progress)
FuzzyClusteringRegulSh (in progress)
FuzzyClusteringRegulQuad (in progress)
"""

import numpy as np
import lib.algorithm as al


class FuzzyClusteringPoly(al.FuzzyClustering):
    """
    FuzzyClusteringPoly(dataset, p, nc)

    A FuzzyClusteringPoly object summarises a fuzzy c-means
    classification, with polynomial fuzzifer function.
    It works like a FuzzyClustering object, storing the dataset,
    performing the algorithm basic steps, and storing the resulting
    partition.

    Notes
    -----
    The polynomial fuzzifier function creates an area of crisp
    clustering around cluster centroids.
    The fuzzifier parameter must be between 0 and 1.

    Parameters
    ----------
    dataset (2d array)
        The dataset to classify. Samples in row, features in columns.
    p (float)
        Value of the fuzzifier.
    nc (integer)
        Number of clusters.

    Attributes
    ----------
    Inherited from FuzzyClustering class.

    Methods
    -------
    _f
        Membership score modification function (fuzzification function)
        used to evaluate the objective function and update the position
        of cluster centroids.
    calculate_memberships
        Calculates the membership scores from the distances between
        data points and cluster centroids, and fuzzifier value.
    """
    min_fuzzifier = 0
    max_fuzzifier = 1

    def _f(self):
        """
        Membership score modification function (fuzzification function)
        used to evaluate the objective function and update the position
        of cluster centroids.
        """
        p = self.fuzzifier
        return (((1 - p) / (1 + p)) * self.memberships ** 2
                + (2 * p / (1 + p)) * self.memberships
                )

    def calculate_memberships(self):
        """
        Calculates membership scores from the distances between samples
        and cluster centroids, and fuzzifier value.

        Notes
        -----
        The polynomial fuzzifier function creates areas of crisp
        clustering around cluster centroids. Some membership scores may
        tend to 0 as a result. They need to be accounted for.
        """
        ct = self.clusters[-1]
        p = self.fuzzifier
        d = np.array([[al.euclidian_distance(i, c)
                       for c in ct
                       ]
                      for i in self.data
                      ]
                     )
        mb = np.zeros(shape=(self.n_samples, self.n_clusters))
        for i in range(self.n_samples):
            # Determine which clusters have non-vanishing membership scores.
            sort_idx = np.argsort(d[i])
            vals = d[i, sort_idx] ** (-2)
            refs = [p/(1 + p*k) * np.sum(d[i, sort_idx[:k+1]] ** (-2))
                    for k in range(self.n_clusters)
                    ]
            comps = [a - b for a, b in zip(vals, refs)]
            lst = [idx for idx, val in enumerate(comps) if val > 0]
            ct_i = max(lst)  # Most distant cluster with non-vanishing score.
            # Get membership scores.
            mb_prime = d[i] ** (-2) - refs[ct_i]
            null_idx = [k for k in range(self.n_clusters)
                        if mb_prime[k] < 0
                        ]
            mb_prime[null_idx] = 0
            mb[i] = [mb_prime[k] / sum(mb_prime)
                     for k in range(self.n_clusters)
                     ]
        # Correct scores to prevent the accumulation of
        # floating point operation errors
        for i in range(self.n_samples):
            mb[i] /= sum(mb[i])
        self.memberships = mb
        return


class FuzzyClusteringRegulSh(al.FuzzyClustering):
    """
    FuzzyClusteringRegulSh(dataset, p, nc)

    A FuzzyClusteringRegulSh object summarises a fuzzy c-means
    classification, with membership regularisation based on Shannon
    entropy.
    It works like a FuzzyClustering object, storing the dataset,
    performing the alogorithm basic steps, and storing the resulting
    partition.

    Notes
    -----
    Instead of using the fuzzifier to soften the classification,
    membership regularisation adds an entropy term to the objective
    function to draw the partition away from crisp clustering.
    Shannon entropy always results in a graded clustering.
    The fuzzifier parameter must be larger than 0.

    Parameters
    ----------
    dataset (2d array)
        The dataset to classify. Samples in row, features in columns.
    p (float)
        Value of the fuzzifier.
    nc (integer)
        Number of clusters.

    Attributes
    ----------
    Inherited from FuzzyClustering class.

    Methods
    -------
    _f
        Membership score modification function (fuzzification function)
        used to evaluate the objective function and update the position
        of cluster centroids.
    _g
        Entropy term used to evaluate the objective function.
    calculate_memberships
        Calculates the membership scores from the distances between
        data points and cluster centroids, and fuzzifier value.
    """
    min_fuzzifier = 0

    def _f(self):
        """
        Membership score modification function (fuzzification function)
        used to evaluate the objective function and update the position
        of cluster centroids.
        """
        return self.memberships

    def _g(self):
        """
        Entropy term used to evaluate the objective function.
        """
        # Shannon entropy always result in graded partitions.
        # The entropy term gives nans if a membership score equals 0.
        return self.fuzzifier * self.memberships * np.log(self.memberships)

    def calculate_memberships(self):
        """
        Calculates membership scores from the distances between samples
        and cluster centroids, and fuzzifier value.

        WARNING: if the spread of the dataset is too large, the exp
        returns 0 and all membership scores are nan.
        Threshold for issue is -d[s]**2/2 ~= 30.
        """

        ct = self.clusters[-1]
        p = self.fuzzifier
        d = np.array([[al.euclidian_distance(i, c)
                       for c in ct
                       ]
                      for i in self.data
                      ]
                     )
        S = 1 / np.sum(np.exp(-d ** 2 / p), axis=1)
        mb = np.array([S[i] * np.exp(-d[i] ** 2 / p)
                       for i in range(self.n_samples)
                       ]
                      )
        # Correct scores to prevent the accumulation of
        # floating point operation errors
        for i in range(self.n_samples):
            mb[i] /= sum(mb[i])
        self.memberships = mb
        return


class FuzzyClusteringRegulQuad(al.FuzzyClustering):
    """
    FuzzyClusteringRegulQuad(dataset, p, nc)

    A FuzzyClusteringRegulQuad object summarises a fuzzy c-means
    classification, with membership regularisation based on quadratic
    entropy.
    It works like a FuzzyClustering object, storing the dataset,
    performing the alogorithm basic steps, and storing the resulting
    partition.

    Notes
    -----
    The quadratic entropy FCM may create null membership scores.
    The fuzzifier parameter must be larger than 0.

    Parameters
    ----------
    dataset (2d array)
        The dataset to classify. Samples in row, features in columns.
    p (float)
        Value of the fuzzifier.
    nc (integer)
        Number of clusters.

    Attributes
    ----------
    Inherited from FuzzyClustering class.

    Methods
    -------
    _f
        Membership score modification function (fuzzification function)
        used to evaluate the objective function and update the position
        of cluster centroids.
    _g
        Entropy term used to evaluate the objective function.
    calculate_memberships
        Calculates the membership scores from the distances between
        data points and cluster centroids, and fuzzifier value.
    """
    min_fuzzifier = 0

    def _f(self):
        """
        Membership score modification function (fuzzification function)
        used to evaluate the objective function and update the position
        of cluster centroids.
        """
        return self.memberships

    def _g(self):
        """
        Entropy term used to evaluate the objective function.
        """
        return self.fuzzifier * self.memberships ** 2

    def calculate_memberships(self):
        """
        Calculates membership scores from the distances between samples
        and cluster centroids, and fuzzifier value.

        Notes
        -----
        The quadratic entropy membership regularisation creates
        vanishing membership scores, that need to be accounted for.
        """
        ct = self.clusters[-1]
        p = self.fuzzifier
        d = np.array([[al.euclidian_distance(i, c)
                       for c in ct
                       ]
                      for i in self.data
                      ]
                     )
        mb = np.zeros(shape=(self.n_samples, self.n_clusters))
        for i in range(self.n_samples):
            # Find clusters with non-vanishing membership scores.
            sort_idx = np.argsort(d[i])
            vals = [np.sum(d[i, sort_idx[:k+1]] ** 2)
                    for k in range(self.n_clusters)
                    ]
            refs = [(k + 1) * d[i, sort_idx[k]] - 2 * p
                    for k in range(self.n_clusters)
                    ]
            comps = [a - b for a, b in zip(vals, refs)]
            lst = [idx for idx, val in enumerate(comps) if val > 0]
            c_i = max(lst)  # Most distant cluster with non-vanishing score.
            # Get membership scores.
            mb[i] = ((1 / c_i)
                     * (1 + np.sum(d[i, sort_idx[:c_i+1]]**2))
                     / (2 * p)
                     - d[i] / (2 * p)
                     )
            null_idx = [k for k in range(self.n_clusters)
                        if mb[i, k] < 0
                        ]
            mb[i, null_idx] = 0
        # Correct scores to prevent the accumulation of
        # floating point operation errors
        for i in range(self.n_samples):
            mb[i] /= sum(mb[i])
        self.memberships = mb
        return
