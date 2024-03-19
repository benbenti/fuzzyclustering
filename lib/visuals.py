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
It may be necessary for further analyses to harmonise the cluster labels across
all the realisations of a clustering solution. We provide a tool to do so.
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
harmonise_cluster_order
typicality
plot_typicality
triangular_gradation_plot
partition_comparison
PCA_plot
tSNE_plot
cluster_update_plot
"""

import umap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import lib.algorithms as al
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter
import pylotwhale.utils.plotTools as pT

colour_set = np.array(["#000000", "#009292", "#004949", "#ff6db6",
                       "#ffb6db", "#490092", "#b66dff", "#006ddb",
                       "#6db6ff", "#b6dbff", "#920000", "#924900",
                       "#db6d00", "#24ff24", "#ffff6d"
                       ]
                      )

plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.linewidth'] = 2.5

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

    dist_data = np.array([[al.euclidian_distance(i, j)
                           for i in FC.data
                           ]
                          for j in FC.data
                          ]
                         )
    dist_clusters = np.array([[al.euclidian_distance(c1, c2)
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


def identify_stable_solutions(dict_FC, plot=False, fileName=None, res=150):
    """
    Finds the optimal number of clusters for each value of fuzzifier,
    and optionally plots the optimal number of clusters as a function
    of the fuzzifier value.
    Clustering solutions which are stable over a large range of
    fuzzifier values are considered to reflect an underlying structure
    in the dataset.

    Arguments
    ---------
    dict_FC (dict)
        The results of a fuzzy c-means classification arranged in a
        dictionary. See module algorithms for details.
    plot (boolean)
        Whether or not to show the plot. Default is False and does not
        display the plot.
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
    the fuzzifier value (optional).
    """

    stable_solutions = np.empty(shape=(len(dict_FC.keys()), 4))
    # The keys of dict_FC are the fuzzifier values.

    # Identify the best clustering solution for each value of fuzzifier.
    for i, p in enumerate(dict_FC.keys()):
        lst_FC = [dict_FC[p][key] for key in dict_FC[p].keys()]
        # Same fuzzifier, different number of clusters.
        quality, idx = al.compare_quality(lst_FC)
        # Uses the same quality index as the fuzzy c-means process.
        best = lst_FC[idx[0]]  # Best fuzzy partition.
        stable_solutions[i] = [p, list(dict_FC[p].keys())[idx[0]],
                               best.n_clusters,
                               quality[idx[0]]
                               ]

    # Plot the optimal number of clusters relative to the fuzzifier value.
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(stable_solutions[:, 0], stable_solutions[:, 2], 'kx')
    plt.xlabel('Fuzzifier value')
    plt.ylabel('Optimal number of clusters')
    plt.ylim(0, max(stable_solutions[:, 2] + 1))
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()
    return stable_solutions


def harmonise_cluster_order(FC, nclust):
    """
    A fuzzy clustering solution is a set of similar dataset partitions.
    The number of clusters is the same for all partitions, and we expect to
    find the same clusters in all realisations. However, the order of the
    clusters in each partition is random.
    In order the summarise the fuzzy clustering solution in a single partition,
    we need to harmonise the cluster order across realisations.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ISSUE: Buggy harmonisation - some clusters may be omitted and others may
    be duplicated, because the ordering method does not warrant a 1-1 asso-
    ciation between clusters. Added a safeguard which cancels harmonisation
    in problematic cases.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Arguments
    ---------
    fc (FuzzyClustering instance)
        The fuzzy partition to harmonise.
    nclust (integer)
        The number of clusters in the solution to harmonise.

    Results
    -------
    No output, the FuzzyClustering instance is updated.
    """

    # Get fuzziness values and initial number of clusters for all realisations
    # of the clustering solution of interest
    sol = identify_stable_solutions(FC)
    pk = [tuple([i[0], i[1]]) for i in sol if i[2] == nclust]

    # Get main cluster (highest membership score) for all realisations
    keys1 = list(FC.keys())
    keys2 = list(FC[keys1[0]].keys())
    nsamples = FC[keys1[0]][keys2[0]].n_samples
    mc = np.zeros(shape=(nsamples, len(pk)), dtype=float)
    for (i, (p, k)) in enumerate(pk):
        typ = typicality(FC[p][k])
        mc[:, i] = typ[:, 1]

    # Cluster harmonisation :
    # Use first realisation as reference
    # For each realisation :
    #  - check which cluster is most similar to reference cluster
    #  - store harmonised order
    order = np.zeros(shape=(len(pk), nclust), dtype=int)
    order[0] = np.arange(nclust)
    ref = mc[:, 0]
    for r in range(1, len(pk)):
        tmp = mc[:, r]
        for i in range(nclust):
            c = Counter(tmp[ref == i])
            order[r, i] = c.most_common()[0][0]

    # Check for duplicates in the order list (= issues with the harmonisation).
    flag = False
    for lst in order:
        s = set(lst)
        if len(s) != len(lst):  # cluster duplication!
            flag = True

    if not flag:
        for (i, (p, k)) in enumerate(pk):
            # Change cluster order.
            ct = FC[p][k].clusters[-1]  # load final cluster coordinates.
            tmp = np.array(ct[order[i], :])  # reorder clusters.
            FC[p][k].clusters.append(tmp)
            # Update column order in membership score table
            mb = FC[p][k].memberships
            tmp = np.array(mb[:, order[i]])
            FC[p][k].memberships = tmp

    return


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


def plot_typicality(FC, grouping=None,
                    fig=None, ax=None, plot=False, fileName=None, res=150
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
    fig(figure)
        Existing figure object to draw the plot, Default, is None and creates
        a new figure
    ax(Axes)
        Existing Axes object to draw the plot. Default is None and creates a
        new plot.
    plot (boolean)
        Whether to display the figure or not. Default is False and does not
        display the histogram.*
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

    typ = typicality(FC)
    if grouping is None:  # Use cluster with highest score to colour samples.
        grouping = typ[:, 1]

    # Sort typicality values by grouping category for colouring.
    typ_lst = [[typ[i, 0] for i in range(len(typ))
                if grouping[i] == group
                ]
               for group in set(grouping)
               ]
    if not fig and not ax:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    plt.hist(typ_lst, bins=20, histtype='barstacked',
             color=colour_set[0:len(set(grouping))]
             )
    plt.xlim(0, 1)
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    if plot:
        plt.show()

    return


def triangular_gradation_plots(FC, c1, c2,
                               restrict=False, cols=['b', 'k'],
                               fig=None, ax=None, plot=True, lgd=True,
                               fileName=None, res=150):
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
    restrict (bool)
        Whether to restrict the triangular plot to samples belonging mostly to
        the displayed clusters. Default is False.
    cols (list)
        2-item list containing colours to use to mark main cluster on graph.
        Default uses blue and black.
    fig(figure)
        Existing figure object to draw the plot, Default, is None and creates
        a new figure
    ax(Axes)
        Existing Axes object to draw the plot. Default is None and creates a
        new plot.
    plot (boolean)
        Whether to show the plot or not. Default is True and shows the plot.
    lgd (bool)
        Whether to label the axes on the plot.
        Default is True and show the labels
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

    # Get main cluster for colouring.
    main_cluster = np.argmax(FC.memberships, axis=1)
    second_cluster = np.argsort(FC.memberships, axis=1)[:, -2]
    c1_ind = main_cluster == c1  # Colour 1
    c2_ind = main_cluster == c2  # Colour 2
    if restrict:  # Select samples which main clusters are c1 and c2.
        ind = [i for i, elt in enumerate(zip(main_cluster, second_cluster))
               if c1 in elt and c2 in elt
               ]
        # Correct choice of indices for colours 1 and 2.
        c1_ind_r = [i for i in ind if c1_ind[i]]
        c2_ind_r = [i for i in ind if c2_ind[i]]
        # Membership scores to plot.
        x_mb_c1 = FC.memberships[c1_ind_r, c1]
        y_mb_c1 = FC.memberships[c1_ind_r, c2]
        x_mb_c2 = FC.memberships[c2_ind_r, c1]
        y_mb_c2 = FC.memberships[c2_ind_r, c2]
    else:
        # Membership scores to plot.
        x_mb_c1 = FC.memberships[c1_ind, c1]
        y_mb_c1 = FC.memberships[c1_ind, c2]
        x_mb_c2 = FC.memberships[c2_ind, c1]
        y_mb_c2 = FC.memberships[c2_ind, c2]
        # Prepare a third colour for remaining samples.
        others = [not (a or b) for a, b in zip(c1_ind, c2_ind)]
        x_mb_oth = FC.memberships[others, c1]
        y_mb_oth = FC.memberships[others, c2]
    if not fig and not ax:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    # Scatter plots, coloured by main clusters.
    if not restrict:
        plt.plot(x_mb_oth, y_mb_oth, color="grey", marker='+', linewidth=0)
    plt.plot(x_mb_c1, y_mb_c1, color=cols[0], marker='+', linewidth=0)
    plt.plot(x_mb_c2, y_mb_c2, color=cols[1], marker='+', linewidth=0)
    # Adjust plot zone.
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # Remove plot frame.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add diagonal lines to the plot.
    plt.plot([0, 1], [1, 0], 'k-', linewidth=0.75)
    plt.plot([0, 0.5], [0, 0.5], 'k-', linewidth=0.75)
    if lgd:
        plt.xlabel('Membership to cluster {}'.format(c1))
        plt.ylabel('Membership to cluster {}'.format(c2))
        plt.title('Gradation between clusters {} and {}'.format(c1, c2))
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    if plot:
        plt.show()
#    else:
#        plt.close()
    return


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
    norm_matrix = corr_matrix / column_sums
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
    pT.display_numbers(fig, ax, corr_matrix, fontSz=8
                       )
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    plt.show()
    return corr_matrix


def PCA_plot(FC, grouping=None, n_std=1,
             fig=None, ax=None, fileName=None, plot=False, res=150
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
        Default is None and uses the fuzzy clusters. Use numeric
        values for grouping categories!
    n_std (integer):
        The number of standard deviations to use in the ellipses.
        1 corresponds roughly to a 68% confidence interval.
        2 corresponds roughly to a 95% confidence interval.
        3 corresponds roughly to a 99.7% confidence interval.
    fig (figure)
        An existing figure object to draw the plot. Default is None and creates
        a new figure.
    ax (Axes)
        An existing Axes object to draw the plot. Default is None and creates a
        new Axes object.
    fileName (path):
        Location to save the figure. Default is None and does not save
        the figure.
    plot (boolean):
        Whether to show the plot or not. Default is False and does not plot the
        figure.
    keep_axes (boolean):
        Whether to return the Axes object of the figure for external use.
        Default is False and does not return the Axes object.
    res (integer)
        Resolution (in dpi) of the saved figure. Default is 150.

    Plots a two-dimensional representation of the data using the first
    two principal components and confidence ellipses corresponding to
    each fuzzy cluster (or other input partition).

    Notes:
    ------
    This function uses code from "https://carstenschelp.github.io/2018/09/14/
    Plot_Confidence_Ellipse_001.html". The figures given for the confidence
    interval do not account for covariance.
    A larger number of outliers should be expected.
    """
    if grouping is None:  # Uses the fuzzy clusters.
        grouping = np.argmax(FC.memberships, axis=1)
        lst_grp = [i for i in range(FC.n_clusters)]
        plot_clusters = True
    else:
        lst_grp = sorted(list(set(grouping)))
        plot_clusters = False
    # Run the PCA and compute the two first principal components.
    pca = PCA(n_components=2, svd_solver='full')
    princ_comp = pca.fit_transform(FC.data)
    if plot_clusters:
        ct_trans = pca.transform(FC.clusters[-1])
    # Make the plot.
    if not fig and not ax:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    for i, grp in enumerate(lst_grp):
        idx = [j for j, elt in enumerate(grouping) if elt == grp]
        x = princ_comp[idx, 0]
        y = princ_comp[idx, 1]
        if len(x) > 0:  # Make scatter plot.
            ax.scatter(x, y, c=colour_set[i], marker='+', label='_nolabel_')
        if len(x) >= 2:  # Draw confidence ellipse.
            cov = np.cov(x, y)
            pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
            # Get the ellipse radius.
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
    if plot_clusters:
        for ind, clust in enumerate(ct_trans):
            ax.plot(clust[0], clust[1], 'x',
                   markersize=10, markeredgewidth=3,
                   c=colour_set[ind], label='_nolabel_'
                   )
    plt.legend(ax.patches, sorted(list(set(grouping))), loc='best')
    ax.set_xlabel('1st principal component - {}% of variance'.format(
                  int(100*pca.explained_variance_ratio_[0]))
                  )
    ax.set_ylabel('2nd principal component - {}% of variance'.format(
                  int(100*pca.explained_variance_ratio_[1]))
                  )
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()
    return


def tSNE_plot(FC, p, n, rs=None, ellipse=False,
              n_cp=None,
              grouping=None,
              fileName=None, res=150,
              plot=False
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
    rs (object):
        Random state to use for the t-SNE algorithm. Default is None and
        chooses a random state based on system and current time.
    ellipse (bool):
        Whether to draw confidence ellipses for each grouping category.
        Default if False and does not draw confidence ellipses.
    n_cp (integer):
        Number of principal components to keep before performing the
        t-SNE.
        Default is None and does not perform the PCA prior to the t-SNE.
        -1 keeps all components.
    grouping (list):
        The grouping categories to colour the data points.
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
    if grouping is None:  # Uses the fuzzy clusters.
        grouping = np.argmax(FC.memberships, axis=1)
    if n_cp is None:  # No data transformation.
        tsne_data = FC.data
    else:  # PCA transformation.
        pca = PCA(n_components=n_cp, svd_solver='full')
        tsne_data = pca.fit_transform(FC.data)
    tsne = TSNE(perplexity=p, n_iter=n, random_state=rs, method='exact')
    plot_data = tsne.fit_transform(tsne_data)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, grp in enumerate(sorted(list(set(grouping)))):
        x = plot_data[grouping == grp, 0]
        y = plot_data[grouping == grp, 1]
        if len(x) > 0:
            ax.scatter(x, y, c=colour_set[i], marker='+')
        if ellipse and len(x) > 2:  # Draw confidence ellipse.
            cov = np.cov(x, y)
            pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
            # Get the ellipse radius.
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
            # Scale to 1 standard deviations and center on the means.
            scale_x = np.sqrt(cov[0, 0]) * 1
            scale_y = np.sqrt(cov[1, 1]) * 1
            # Rotate and scale the ellipse.
            transf = transforms.Affine2D() \
                               .rotate_deg(45) \
                               .scale(scale_x, scale_y) \
                               .translate(np.mean(x), np.mean(y))
            ellipse.set_transform(transf + ax.transData)
            ax.add_patch(ellipse)
    plt.legend(sorted(list(set(grouping))),
               bbox_to_anchor=(1.05, 1),
               loc='upper left'
               )
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close('all')

    return


def UMAP_plot(FC, n_neighbors=75, min_dist=0.75,
              grouping=None,
              fig=None, ax=None, plot=True, fileName=None, res=150):
    """
    Runs the UMAP algorithm to represent the dataset in a 2D plot.
    The UMAP projection conserves the similarity between datapoints, which
    means that the closer two samples are in the UMAP plot, the more similar
    they are.

    Arguments:
    ----------
    FC (FuzzyClustering instance):
        Contains the dataset and the fuzzy partition of the data.
    grouping (list):
        The grouping categories to plot the confidence ellipses.
        Default is None and uses the fuzzy clusters. Use numeric
        values for grouping categories!
    fig (figure)
        An existing figure object to draw the plot. Default is None and creates
        a new figure.
    ax (Axes)
        An existing Axes object to draw the plot. Default is None and creates a
        new Axes object.
    fileName (path):
        Location to save the figure. Default is None and does not save
        the figure.
    plot (boolean):
        Whether to show the plot or not. Default is False and does not plot the
        figure.
    res (integer)
        Resolution (in dpi) of the saved figure. Default is 150.
    """
    if grouping is None:  # Uses the fuzzy clusters.
        grouping = np.argmax(FC.memberships, axis=1)
        lst_grp = [i for i in range(FC.n_clusters)]
    else:
        lst_grp = sorted(list(set(grouping)))
    # Run the UMAP and compute the 2D projection.
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    proj = reducer.fit_transform(FC.data)
    # Make the plot.
    if not fig and not ax:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    for i, grp in enumerate(lst_grp):
        idx = [j for j, elt in enumerate(grouping) if elt == grp]
        x = proj[idx, 0]
        y = proj[idx, 1]
        if len(x) > 0:  # Make scatter plot.
            ax.scatter(x, y, c=colour_set[i], marker='o', label=grp)
    ax.set_xlabel('1st UMAP dimension')
    ax.set_ylabel('2nd UMAP dimension')
    plt.legend(loc='best')
    if fileName is not None:
        plt.savefig(fileName, dpi=res, bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()
    return


def cluster_update_plot(FC, start=None, stop=None, fig=None, ax=None):
    """Makes a PCA plot which shows cluster updates.

    Arguments
    ---------
    FC (FuzzyClustering instance)
    start (int):
        which loop number to start the plot with. Solves cluster fusion issues.
    stop (int)
        which loop number to end the plot with. Solves cluster fusion issues.
    fig, ax (matplotlib.pyplot.Figure and Axes instances):
        existing figure to plot in.

    Returns
    -------
    A plot with the successive position of the fuzzy clusters
    """

    # Run the PCA and compute the two first principal components.
    pca = PCA(n_components=2, svd_solver='full')
    princ_comp = pca.fit_transform(FC.data)
    # Plot samples in grey.
    if not fig and not ax:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
    x = princ_comp[:, 0]
    y = princ_comp[:, 1]
    ax.scatter(x, y, c="#bdbdbd", marker='+', label='_nolabel_')
    ax.set_xlabel('1st principal component - {}% of variance'.format(
                  int(100*pca.explained_variance_ratio_[0]))
                  )
    ax.set_ylabel('2nd principal component - {}% of variance'.format(
                  int(100*pca.explained_variance_ratio_[1]))
                  )
    # Plot successive cluster positions
    # Compute cluster positions along principal components.
    pos = []
    for i in FC.clusters:
        tmp = pca.transform(i)
        pos.append(tmp)
    # Remove loops before start and after stop.
    pos = pos[start:stop]  # Slices from start/until stop if None.
    # Plot cluster movement with loops.
    for i in range(FC.n_clusters):
        x = [k[i, 0] for k in pos]
        y = [k[i, 1] for k in pos]
        if len(x) > 1:
            plt.plot(x, y, 'x-', c=colour_set[i])
        # Highlight last cluster position.
        plt.plot(x[-1], y[-1], 'x', c=colour_set[i],
                 markersize=10, markeredgewidth=2.5
                 )
    plt.show()
    return
