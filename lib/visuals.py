"""
This module implements tools to visualise and analyse the results of a
fuzzy c-means (FCM) classification.
-----------------------------------------------------------------------
The distance checking tool checks the distances between cluster
centroids against the distances between data points to verify how
cluster centroids are distributed across the feature space.
The FCM classification is run for a wide range of fuzzifier values and
number of clusters. The first step of the analysis of FCM results is to
identify interesting clustering solutions. Clustering solutions which
are stable over a wide range of fuzzifier values are considered
representative of an underlying structure.
To further characterise the fuzzy partitions, this module propose the
computation of the typicality of samples - a quantitative measure of
whether samples are much closer to one cluster than the others or how
they lie between categories - and a visualisation method for
typicality.
In addition, triangular plots, with the membership to cluster A on the
x-axis and membership to cluster B on the y-axis, display where graded
samples lie between clusters.
Moreover, if another (crisp) partition of the dataset is available (as
ground truth or for method comparison), this module proposes to compare
the partitions, by assigning each sample to the cluster to which it has
the highest membership score and displaying a correspondance matrix.
Finally, this module can represent high-dimensional datasets in a 2D
space, using either a PCA or t-SNE approach, to visualise the
distribution of fuzzy clusters and samples in the feature space.

-----------------------------------------------------------------------
FUNCTIONS
---------
distance_check
identify_stable_solutions
typicality
plot_typicality
triangular_gradation_plot
partition_comparison
PCA_plot
tSNE_plot
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import lib.algorithms as algo
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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
    colours[0]) and pairwise distances between cluster centroids (in
    colours[1]).
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
    Finds the optimal number of clusters for each value of fuzzifier,
    and plots the optimal number of clusters as a function of the
    fuzzifier value.
    Clustering solutions which are stable over a large range of
    fuzzifier values are considered to reflect an underlying structure
    in the dataset.

    Arguments
    ---------
    dict_FC (dict)
        The results of a fuzzy c-means classification arranges in a
        dictionary. See module algorithms for details.
    fileName (path)
        Location to save the figure in. Default is None and does not
        save the figure.
    res (integer)
        Resolution, in dpi, of the saved figure. Default is 150.

    Returns
    -------
    stable_solutions (2d array)
        A three column array containing the fuzzifer values (1st col),
        the optimal number of clusters (2nd col) and the quality index
        of the optimal solution (3rd col).
    Plots the number of clusters in the best solution as a function of
    the fuzzifier value.
    """

    stable_solutions = np.empty(shape=(len(dict_FC.keys()), 3))
    # The keys of dict_FC are the fuzzifier values.

    # Identify the best clustering solution for each value of fuzzifier.
    for i, p in enumerate(dict_FC.keys()):
        lst_FC = [dict_FC[p][key] for key in dict_FC[p].keys()]  # Same fuzzifier, different number of clusters.
        quality, idx = [elt.quality for elt in lst_FC]  # Uses the same quality index as the fuzzy c-means process.
        best = lst_FC[idx[0]]  # Best fuzzy partition.
        stable_solutions[i] = [p, best.n_clusters, quality[idx[0]]]

    # Plot the optimal number of clusters relative to the fuzzifier value.
    plt.plot(stable_solutions[:, 0], stable_solutions[:, 1], 'kx')
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
        Samples in rows. 1st column: typicality, 2nd column: cluster
        with the highest membership score, 3rd column: cluster with the
        second highest membership score.
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


def plot_typicality(FC, grouping=None, colour_set=None,
                    fileName=None, res=150
                    ):
    """
    Plots a stacked histogram of typicality values, coloured by main
    cluster or by another provided partition.

    Arguments
    ---------
    FC (FuzzyClustering instance)
        The fuzzy partition to plot.
    grouping (list)
        The grouping categories used to colour the samples in the
        histogram. Default is None and colour the samples by fuzzy
        cluster with the highest membership score.
    colour_set (list)
        The list of colours to use in the histogram. Default is None
        and uses a default set of 13 colourblind-suitable colours.
    fileName (path)
        Location to save the figure. Default is None and does not save
        the figure.
    res (integer)
        Resolution (in dpi) of the save figure. Default is 150 dpi.

    Returns
    -------
    An histogram of typicality values. The samples are coloured by
    category and categories are stacked.
    """

    mb = FC.memberships
    typ = typicality(mb)
    if grouping is None:  # Use cluster with highest score to colour samples.
        grouping = typ[:, 1]

    # Sort typicality values by grouping category for colouring.
    typ_lst = [[typ[i, 0] for i in range(len(typ))
                if grouping[i] == group
                ]
               for group in set(grouping)
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
    plt.plot(x_mb, y_mb, 'b+')
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
    (as clusters with the highest membership scores) and another crisp
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
        Location to save the figure. Default is None and does not
        save the figure.
    res (integer)
        The resolution (in dpi) of the saved figure.
        Default is 150 dpi.

    Returns
    -------
    corr_matrix (2d array)
        A correspondence matrix between the two partition. Each
        element of the matrix is the count of samples which belong
        to a given pair of fuzzy cluster and partition category.
    Draws the correspondence matrix.
    """

    main_cluster = np.argmax(FC.memberships, axis=1)
    corr_matrix = np.zeros(shape=(FC.n_clusters, len(set(partition))))
    for i in range(FC.n_samples):
        corr_matrix[main_cluster[i], partition[i]] += 1
        # Adds each sample to the count of the corresponding category.
    column_sums = np.sum(corr_matrix, axis=0)
    norm_matrix = corr_matrix / column_sums.
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
    ax.set_xlabel('Reference partition')
    ax.set_ylabel('Fuzzy clusters')
    # Display sample counts in the image.
    pT.display_numbers(fig, ax, corr_matrix,
                       condition=None, fontSz=8
                       )
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    plt.show()
    return corr_matrix


def PCA_plot(FC, grouping=None, n_std=1, colour_set=None,
             fileName=None, res=150
             ):
    """
    Plots the dataset using the first two dimensions of a PCA. Adds a
    confidence ellipse for each fuzzy cluster (or another partition).

    Arguments:
    ----------
    FC (FuzzyClustering instance):
        Contains the dataset and the fuzzy partition of the data.
    grouping (list):
        The grouping categories to plot the confidence ellipses.
    n_std (integer):
        The number of standard deviations to use in the ellipses.
        1 corresponds roughly to a 68% confidence interval.
        2 corresponds roughly to a 95% confidence interval.
        3 corresponds roughly to a 99.7% confidence interval.
    colour_set (list):
        The colours used to represent the different categories in the
        dataset and their confidence ellipse. Default is None and uses
        a predefined colourblind-friendly set of 13 colours.
    fileName (path):
        Location to save the figure. Default is None and does not save
        the figure.
    res (integer)
        Resolution (in dpi) of the saved figure. Default is 150.

    Plots a two-dimensional representation of the data using the first
    two principal components and confidence ellipses corresponding to
    each fuzzy cluster (or other input partition).

    Notes:
    ------
    This function uses code from https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    The figures given for the confidence interval do not account for
    covariance. A larger number of outliers should be expected.
    """
    if colour_set is None:
        colour_set = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a',
                      '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f',
                      '#cab2d6', '#ffff99', '#ffffff'
                      ]
    if grouping is None:  # Uses the fuzzy clusters.
        grouping = np.argmax(FC.memberships, axis=1)
    # Compute the two first principal components.
    pca = PCA(n_components=2)
    pc = pca.fit_transform(FC.data)
    fig, ax = plt.subplots()
    for grp, i in enumerate(set(grouping)):
        x = pc[grouping == grp, 0]
        y = pc[grouping == grp, 1]
        ax.scatter(x, y, c=colour_set[i], marker='+')
        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Get the ellipse radii.
        radius_x = np.sqrt(1 + pearson)
        radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=radius_x * 2,
                          height=radius_y * 2,
                          facecolor=colour_set[i],
                          alpha=0.3,
                          edgecolor=colour_set[i],
                          linewidth=2
                          )
        # Scale to n_std standard deviations and center on the means.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        # Rotate and scale the ellipse.
        transf = transforms.Affine2D() \
                 .rotate_deg(45) \
                 .scale(scale_x, scale_y) \
                 .translate(np.mean(x), np.mean(y))
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    plt.show()


def tSNE_plot(FC, p, n,
              n_cp=None,
              grouping=None, colour_set=None,
              fileName=None, res=150
              ):
    """
    Plots the dataset in 2D using the t-SNE algorithm. Colours data
    points according to fuzzy cluster (or other grouping categories).

    Arguments:
    ----------
    FC (FuzzyClustering instance):
        Contains the dataset and the fuzzy partition of the data.
    p (integer):
        Perplexity. Number of closest neighbours to use in the t-SNE.
        Lower values focus on local structure, larger values on overall
        structure. Usual values in [5, 50].
        Should not exceed the number of data points.
    n (integer):
        Number of iterations to perform for the t-SNE.
        Usual values in [1000-5000].
    n_cp (integer):
        Number of principal components to keep before performing the
        t-SNE.
        Default is None and does not perform the PCA prior to the t-SNE.
        -1 keeps all components.
    grouping (list):
        The grouping categories to colour the data points.
    colour_set (list):
        The colours used to represent the different categories in the
        dataset and their confidence ellipse. Default is None and uses
        a predefined colourblind-friendly set of 13 colours.
    fileName (path):
        Location to save the figure. Default is None and does not save
        the figure.
    res (integer)
        Resolution (in dpi) of the saved figure. Default is 150.

    Plots a two-dimensional representation of the data using the t-SNE
    algorithm. The data points are coloured according to fuzzy cluster
    (or another partition).

    Notes:
    ------
    It may be necessary to use several values of perplexity to have a
    complete view of the dataset.
    """
    if colour_set is None:
        colour_set = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a',
                      '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f',
                      '#cab2d6', '#ffff99', '#ffffff'
                      ]
    if grouping is None:  # Uses the fuzzy clusters.
        grouping = np.argmax(FC.memberships, axis=1)
    if n_cp is None:  # No data transformation.
        tsne_data = FC.data
    else:  # PCA transformation.
        pca = PCA(n_components=n_cp)
        tsne_data = pca.fit_transform(FC.data)
    tsne = TSNE(perplexity=p, n_iter=n)
    plot_data = tsne.fit_transform(tsne_data)
    fig, ax = plt.subplots()
    for grp, i in enumerate(set(grouping)):
        x = plot_data[grouping==grp, 0]
        y = plot_data[grouping==grp, 1]
        ax.scatter(x, y, c=colour_set[i], marker='+')
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    plt.show()
