"""
This module presents tools to visualise and analyse the results of a
fuzzy c-means (FCM) classification.
-----------------------------------------------------------------------
The distance checking tools checks the distances between cluster
centroids against the distances between data points to verify that
cluster centroids are fairly distributed across the feature space.
The FCM classification is run for a wide range of fuzzifier values and
number of clusters. The first step of the analysis of FCM results is to
identify interesting clustering solutions. Clustering solutions which
are stable over wide range of fuzzifier values are considered
representative of an underlying structure.
To further characterise the fuzzy partitions, this module propose the
computation of the typicality of samples - a quantitative measure of
how samples are closer to one cluster than the other ones - and a
visualisation method for typicality.
In addition, triangular plots - with the membership to cluster A on the
x-axis and membership to cluster B on the y-axis displays where graded
samples lie between clusters.
Finally, if another, crisp, partition of the dataset is available (as
ground truth or for method comparison), this module propose to compare
the partition, by assigning each sample to the cluster to which it has
the highest membership score.

Notes
-----
Make the module compatible with fuzzy partitions as a collection of
arrays is on my to-do list.
At the moment, you'd need to arrange the array in a FuzzyClustering
instance to use the visualisation tools with an external partition.
-----------------------------------------------------------------------
FUNCTIONS
---------
distance_check
identify_stable_solutions
typicality
plot_typicality
triangular_gradation_plot
stereotypical_sample
partition_comparison
"""

import numpy as np
import matplotlib.pyplot as plt

import fuzzyclustering.algorithms as algo
import pylotwhale.MLwhales.MLtools_beta as myML
import pylotwhale.utils.plotTools as pT


def distance_check(FC, colours=['k', 'b'], YLIM=None,
                   fileName=None, res=150
                   ):
    """
    Plots an histogram to compare the pairwise distances between data
    points and the pairwise distances between cluster centroids.

    Parameters
    ----------
    FC (FuzzyClustering instance)
        Fuzzy clustering results.
    colours (2-element list)
        Colours to use for distances between data points (1st element)
        and cluster centroids (2nd element). Default is black and blue.
    YLIM (float)
        Maximum value to display on the y-axis. Allows zooming in if
        the number of clusters is much smaller than the number of
        datapoints. Default is None and does not zoom in.
    fileName (path)
    The location where to save the figure. Default is None and does
        not save the figure.
    res (integer)
        Resolution (in dpi) to use when saving the figure.
        Default is 150.

    Returns
    -------
    An histogram of the pairwise distances between data points (in
    colour[0]) and pairwise distances between cluster centroids (in
    colour[1]).
    """

    dist_data = np.array([[algo.euclidian_distance(i, j)
                           for i in FC.data
                           ]
                          for j in FC.data
                          ]
                         )
    dist_clusters = np.array([algo.euclidian_distance(c1, c2)
                               for c1 in FC.clusters[-1]
                               ]
                              for c2 in FC.clusters[-1]
                              ]
                             )
    plt.hist((dist_data.flat, dist_clusters.flat), histtype='bar',
             color=colours, label=('data points', 'cluster centroids')
             )
    plt.xlabel('Euclidian distances')
    plt.ylabel('Number of occurrences')
    plt.legend('data points', 'clusters')
    if YLIM is not None:
        plt.ylim(0, YLIM)
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    plt.show()


def identify_stable_solutions(dict_FC, fileName=None, res=150):
    """
    Find the optimal cluster solution for each value of fuzzifier, and
    plots the number of clusters in the best solution as a function of
    the fuzzifier value.

    Arguments
    ---------
    dict_FC (dict of dicts)
    fileName (path)
    res (integer)

    Returns
    -------
    stable_solutions (array)
    Plots the number of clusters in the best solution as a function of
    the fuzzifier value.
    """

    stable_solutions = np.empty(shape=(len(dict_FC.keys()), 3))
    # Identify best clustering solution for each value of fuzzifier.
    for i, p in enumerate(dict_FC.keys()):
        lst_FC = [dict_FC[p][key] for key in dict_FC[p].keys()]
        quality, idx = algo.quality_compare(lst_FC)
        best = lst_FC[idx[0]]  # Best fuzzy partition.
        stable_solutions[i] = [p, best.n_clusters, quality(idx[0])]
    # Plot optimal number of clusters relative to the fuzziness value.
    plt.plot(stable_solutions[:, 0], stable_solutions[:, 2], 'kx')
    plt.xlabel('Fuzzifier value')
    plt.ylabel('Optimal number of clusters')
    plt.ylim(0, max(stable_solutions[:, 1] + 1))
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    plt.show()
    return stable_solutions


def typicality(FC):
    """
    Measures the typicality of all samples of the dataset. The
    typicality of a sample is the difference between its two highest
    membership scores.

    Arguments
    ---------
    FC (FuzzyClustering instance)
        Fuzzy partition

    Returns
    -------
    typicality (2d array)
        For each sample, its typicality and the indices of the two
        clusters with the highest membership scores.
    """

    mb = FC.memberships
    typicality = np.zeros(shape=(len(mb), 3))
    for i in range(FC.n_samples):
        idx = np.argsort(mb[i])
        typicality[i] = (mb[i, idx[-1]] - mb[i, idx[-2]],
                         idx[-1],
                         idx[-2]
                         )
    return typicality


def plot_typicality(FC, partition=None, colour_set=None,
                    fileName=None, res=150
                    ):
    """
    Plots a stacked histogram of typicality values, coloured by main
    cluster or by another provided partition.

    Arguments
    ---------
    FC (FuzzyClustering instance)
        The fuzzy partition to plot.
    partition (list)
        The grouping categories used to colour the samples in the
        histogram. Default is None and colour the samples by main fuzzy
        cluster (cluster with the highest membership score).
    colour_set (list)
        The list of colours to use in the histogram. Default is None
        and uses a default set of 13 colourblind-suitable colours.
    fileName (path)
        Where to save the figure. Default is None and does not save the
        figure.
    res (integer)
        Resolution (in dpi) to use when saving the figure.
        Default is 150 dpi.

    Returns
    -------
    An histogram of typicality values. The samples are coloured by
    category and categories are stacked.

    Notes
    -----
    The FuzzyClustering argument overrides the array argument for
    memberships.
    """

    mb = FC.memberships
    typ = typicality(mb)
    if partition is None:
        partition = typ[:, 1]  # Use main cluster to colour samples.
    # Sort typicality values by partition for coloring
    typ_lst = [[typ[i, 0] for i in range(len(typ))
                if partition[i] == group
                ]
               for group in set(partition)
               ]

    if colour_set is None:
        colour_set = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a',
                      '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f',
                      '#cab2d6', '#ffff99', '#ffffff'
                      ]
    plt.hist(typ_lst, bins=20, histtype='barstacked', color=colour_set)
    plt.xlim(0, 1)
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    plt.show()


def triangular_gradation_plots(FC, c1, c2, fileName=None, res=150):
    """
    Makes a triangular plot with the membership scores to a first
    cluster on the x-axis and the membership to a second cluster on the
    y-axis.

    Arguments
    ---------
    FC (FuzzyClustering instance)
        The fuzzy partition.
    c1 (integer)
        The index of the first cluster.
    c2 (integer)
        The index of the second cluster.
    fileName (path)
        Where to save the figure. Default is None and does not save
        the figure.
    res (integer)
        The resolution (in dpi) to use when saving the figure.
        Default is 150.

    Returns
    -------
    A triangular gradation plot.
    """

    x_mb = FC.memberships[:, c1]
    y_mb = FC.memberships[:, c2]
    plt.plot(x_mb, y_mb, 'bo')
    plt.xlabel('Membership to cluster {}'.format(c1))
    plt.ylabel('Membership to cluster {}'.format(c2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Gradation between clusters {} and {}'.format(c1, c2))
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    plt.show()


def partition_comparison(FC, partition, fileName=None, res=150):
    """
    Constructs a matrix of correspondence between the fuzzy partition
    (as clusters with the highest membership scores) and another, crisp
    partition of the dataset.

    Arguments
    ---------
    FC (FuzzyClustering instance)
        The fuzzy partition of the dataset.
    partition (list)
        The second (crisp) parition of the dataset. A list of the
        categories to which each sample belongs. Categories should
        be coded by integer values.
    fileName (path)
        Where to save the display. Default is None and does not save
        the display.
    res (integer)
        The resolution (in dpi) to use when saving the display.
        Default is 150 dpi.

    Returns
    -------
    corr_matrix (2d array)
        A correspondence matrix between the two partition. Each
        element of the matrix is the count of samples which belong
        to a given pair of fuzzy cluster and partition category.
    """

    main_cluster = np.argmax(FC.memberships, axis=1)
    corr_matrix = np.zeros(shape=(FC.n_clusters, len(set(partition))))
    for i in range(FC.n_samples):
        corr_matrix[main_cluster[i], partition[i]] += 1

    plt.imshow(corr_matrix, cmap=plt.cm.blues, alpha=0.5)

    norm_matrix = myML.scale(corr_matrix,
                             normfun=myML.colCounts2colFreqs, axis=0
                             )
    fig, ax = plt.subplots()
    ax.imshow(norm_matrix, cmap=plt.cm.Blues, alpha=0.5,
              interpolation='nearest'
              )
    # Tick labels
    ax.set_xticks(range(len(set(partition))))
    ax.set_xticklabels(set(partition), rotation=90)
    ax.set_yticks(range(FC.n_clusters))
    ax.set_yticklabels(range(FC.n_clusters))
    # Axis labels
    ax.set_xlabel('Crisp partition category')
    ax.set_ylabel('Fuzzy cluster')
    pT.display_numbers(fig, ax, corr_matrix,
                       condition=None, fontSz=8
                       )
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    plt.show()

    return corr_matrix
