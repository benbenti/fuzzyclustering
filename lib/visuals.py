"""
This module implements tools to visualise and analyse the results of a
fuzzy c-means (FCM) classification.
-----------------------------------------------------------------------
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
from itertools import combinations
import matplotlib.transforms as transforms
import lib.algorithms as al
from sklearn.decomposition import PCA
from collections import Counter

colour_set = np.array(["#000000", "#009292", "#004949", "#ff6db6",
                       "#ffb6db", "#490092", "#b66dff", "#006ddb",
                       "#6db6ff", "#b6dbff", "#920000", "#924900",
                       "#db6d00", "#24ff24", "#ffff6d"
                       ]
                      )

plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.linewidth'] = 2.5


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
        A four column array containing the fuzzifer values (1st col),
        the fuzziness value (2nd col), the inital and final number of clusters
        (3rd and 4th cols).
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

    if plot or fileName:
        # Plot the optimal number of clusters relative to the fuzzifier value.
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(stable_solutions[:, 0], stable_solutions[:, 2], 'kx')
        ax.xlabel('Fuzzifier value')
        ax.ylabel('Optimal number of clusters')
        ax.ylim(0, max(stable_solutions[:, 2] + 1))
        if fileName is not None:
            plt.savefig(fileName, dpi=res, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()

    return stable_solutions


def harmonise_cluster_order(dict_FC, nclust):
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
    dict_FC (dict)
        The results of a fuzzy c-means classification arranged in a
        dictionary. See module algorithms for details.
    nclust (integer)
        The number of clusters in the solution to harmonise.

    Results
    -------
    No output, the FuzzyClustering instances are updated.
    """

    # Get fuzziness values and initial number of clusters for all realisations
    # of the clustering solution of interest
    sol = identify_stable_solutions(dict_FC)
    pk = [tuple([i[0], i[1]]) for i in sol if i[2] == nclust]

    # Get main cluster (highest membership score) for all realisations
    keys1 = list(dict_FC.keys())
    keys2 = list(dict_FC[keys1[0]].keys())
    nsamples = dict_FC[keys1[0]][keys2[0]].n_samples
    mc = np.zeros(shape=(nsamples, len(pk)), dtype=float)
    for (i, (p, k)) in enumerate(pk):
        typ = typicality(dict_FC[p][k])
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
        if len(s) != len(lst):  # cluster duplication or omission!
            flag = True

    if not flag:
        for (i, (p, k)) in enumerate(pk):
            # Change cluster order.
            ct = dict_FC[p][k].clusters[-1]  # load final cluster coordinates.
            tmp = np.array(ct[order[i], :])  # reorder clusters.
            dict_FC[p][k].clusters.append(tmp)
            # Update membership scores
            mb = dict_FC[p][k].memberships
            tmp = np.array(mb[:, order[i]])
            dict_FC[p][k].memberships = tmp

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


def plot_typicality(FC, grouping=None, plot_type='bar',
                    fig=None, ax=None
                    ):
    """
    Plots an histogram of typicality values, coloured by main
    cluster or by another provided partition.

    Arguments
    ---------
    FC (FuzzyClustering instance)
        The fuzzy partition to plot.
    grouping (list)
        The grouping categories used to colour the samples in the
        histogram. Default is None and colour the samples by fuzzy
        cluster with the highest membership score.
    plot_type (str):
        The type of histogram to plot. Default is 'bar' and displays the
        fuzzy clusters side by side. Other options are 'barstacked',
        'step' and 'stepfilled'. See plt.hist for details.
    fig (figure)
        Existing figure object to draw the plot, Default, is None and creates a
        new figure.
    ax (Axes)
        Existing Axes object to draw the plot. Default is None and creates a
        new plot.

    Returns
    -------
    fig (plt.Figure instance)
    ax (plt.Axes instance):
        An histogram of typicality values with samples coloured by category.
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
        ax = fig.add_subplot()
    ax.hist(typ_lst,
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            histtype=plot_type,
            color=colour_set[0:len(set(grouping))]
            )
    ax.xlim(0, 1)
    plt.close()

    return fig, ax


def triangular_gradation_plot(FC, c1, c2,
                              restrict=False, cols=['b', 'k'],
                              fig=None, ax=None, lgd=True
                              ):
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
    fig (plt.Figure)
        Existing figure object to draw the plot, Default, is None and creates
        a new figure
    ax (plt.Axes)
        Existing Axes object to draw the plot. Default is None and creates a
        new plot.
    lgd (bool)
        Whether to label the axes on the plot.
        Default is True and show the label.

    Returns
    -------
    fig (plt.Figure)
    ax (plt.Axes)
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
        others = [(not a and not b) for a, b in zip(c1_ind, c2_ind)]
        x_mb_oth = FC.memberships[others, c1]
        y_mb_oth = FC.memberships[others, c2]
    if not fig and not ax:
        fig = plt.figure()
        ax = fig.add_subplot()
    # Scatter plots, coloured by main clusters.
    if not restrict:
        ax.plot(x_mb_oth, y_mb_oth, color="grey", marker='+', linewidth=0)
    ax.plot(x_mb_c1, y_mb_c1, color=cols[0], marker='+', linewidth=0)
    ax.plot(x_mb_c2, y_mb_c2, color=cols[1], marker='+', linewidth=0)
    # Adjust plot zone.
    ax.xlim(0, 1)
    ax.ylim(0, 1)
    # Remove plot frame.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add diagonal lines to the plot.
    ax.plot([0, 1], [1, 0], 'k-', linewidth=0.75)
    ax.plot([0, 0.5], [0, 0.5], 'k-', linewidth=0.75)
    if lgd:
        ax.xlabel('Membership to cluster {}'.format(c1))
        ax.ylabel('Membership to cluster {}'.format(c2))
    plt.close()

    return fig, ax


def triangular_gradation_plots(FC, restrict=False,
                               fig=None, plot=True
                               ):
    """
    Makes a compound figure with triangular gradation plots for every cluster
    pair.

    Arguments
    ---------
    FC (FuzzyClustering instance)
        The fuzzy partition.
    restrict (bool)
        Whether to restrict the triangular plot to samples belonging mostly to
        the displayed clusters. Default is False.
    fig(figure)
        Existing figure object to draw the plot, Default, is None and creates
        a new figure.

    Returns
    -------
    fig (plt.Figure)
    ax (plt.Axes)
        A figure with all triangular gradation plots.
    """

    # Prepare figure for plotting.
    if not fig:
        fig = plt.figure(figsize=(10*(FC.n_clusters-1), 10*(FC.n_clusters-1)))
    # Prepare subplot locations.
    fig.add_gridspec(FC.n_clusters-1, FC.n_clusters-1)

    # Loop through cluster pairs and build plots.
    for c1, c2 in combinations(range(FC.n_clusters), 2):
        ax = plt.subplot2grid((FC.n_clusters-1, FC.n_clusters-1),
                              (c2-1, c1), fig=fig
                              )
        # Get the membership scores to both clusters on plot.
        # Sort samples by main cluster for colouring.
        main_cluster = np.argmax(FC.memberships, axis=1)
        second_cluster = np.argsort(FC.memberships, axis=1)[:, -2]
        # Identification of samples belonging mainly to c1 and c2.
        c1_ind = (main_cluster == c1)
        c2_ind = (main_cluster == c2)
        if restrict:
            ind = [i for i, elt in enumerate(zip(main_cluster, second_cluster))
                   if c1 in elt and c2 in elt
                   ]
            # Sort remaining samples by main cluster.
            c1_ind_r = [i for i in ind if c1_ind[i]]
            c2_ind_r = [i for i in ind if c2_ind[i]]
            # Get membership scores to c1 and c2 for both subsets of samples.
            x_mb_c1 = FC.memberships[c1_ind_r, c1]
            y_mb_c1 = FC.memberships[c1_ind_r, c2]
            x_mb_c2 = FC.memberships[c2_ind_r, c1]
            y_mb_c2 = FC.memberships[c2_ind_r, c2]
        else:  # Plot all samples.
            # Get membership score to plot for three subsets of samples:
            # main cluster c1, main cluster c2, remaining samples.
            x_mb_c1 = FC.memberships[c1_ind, c1]
            y_mb_c1 = FC.memberships[c1_ind, c2]
            x_mb_c2 = FC.memberships[c2_ind, c1]
            y_mb_c2 = FC.memberships[c2_ind, c2]
            others = [(not a and not b) for a, b in zip(c1_ind, c2_ind)]
            x_mb_oth = FC.memberships[others, c1]
            y_mb_oth = FC.memberships[others, c2]
        # Make scatter plots coloured by main cluster.
        if not restrict:  # Plot calls with main cluster != c1 or c2
            plt.plot(x_mb_oth, y_mb_oth,
                     color="grey", marker='+', linewidth=0
                     )
        plt.plot(x_mb_c1, y_mb_c1,
                 color=colour_set[c1], marker='+', linewidth=0
                 )
        plt.plot(x_mb_c2, y_mb_c2,
                 color=colour_set[c2], marker='+', linewidth=0
                 )
        # Adjust plot zone.
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # Remove plot frame.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add diagonal lines to the plot.
        plt.plot([0, 1], [1, 0], 'k-', linewidth=0.75)
        plt.plot([0, 0.5], [0, 0.5], 'k-', linewidth=0.75)
        # Add axes labels.
        if c2 == FC.n_clusters-1:  # Last row of subplots.
            ax.set_xlabel("Membership to cluster {}".format(c1))
        if c1 == 0:  # First column of plots.
            ax.set_ylabel("Membership to cluster {}".format(c2))
    plt.close()

    return fig, ax


def make_partition_comparison(FC, partition, v=1):
    """
    Constructs a matrix of correspondence between the fuzzy partition
    (as clusters with the highest membership scores) and another partition
    of the dataset.

    Arguments
    ---------
    FC (FuzzyClustering instance)
        The fuzzy partition of the dataset.
    partition (list)
        The second parition of the dataset. A list of the
        categories to which each sample belongs.
    v (int):
        The computation method for the correspondence table.
        Default is 1 and uses raw membership scores. Any alternative
        uses main cluster for counting.

    Returns
    -------
    corr_matrix (2d array)
        A correspondence matrix between the two partition. Each
        element of the matrix is the count of samples which belong
        to a given pair of fuzzy cluster and partition category.
    """

    # Get fuzzy cluster with highest membership for each sample.
    main_cluster = np.argmax(FC.memberships, axis=1)
    # Make sorted list of categories in the partition.
    cat_list = sorted(list(set(partition)))
    # Initialise correspondence matrix.
    c_mat = np.zeros(shape=(FC.n_clusters, len(cat_list)))
    for i in range(FC.n_samples):  # Loop over samples.
        if v == 1:  # Use raw membership scores in c_mat
            c_mat[:, cat_list.index(partition[i])] += FC.memberships[i]
        else:  # Use main cluster.
            c_mat[main_cluster[i], cat_list.index(partition[i])] += 1

    return c_mat


def plot_partition_comparison(c_mat, name_list, v=1):
    """
    Makes a plot of the correspondence matrix obtained with
    make_partition_comparison.

    Arguments
    ---------
    c_mat (2D numpy array):
        The correspondence matrix between fuzzy clusters and
        a reference partion of the dataset.
    name_list (list):
        The list of category names in the reference partition.
    v (int):
        The computation method for the comparison table.
        Changes slightly the figure details.

    Returns:
    --------
    fig (plt.Figure instance)
    ax (plt.Axes instance)
        The figure with the correspondence matrix
    """

    fig = plt.figure()
    ax = fig.add_subplot()

    # Version 1: Membership scores in the table
    if v == 1:
        col_sums = np.sum(c_mat, axis=0)
        norm_mat = c_mat / col_sums
        ax.imshow(norm_mat, cmap=plt.cm.Blues,
                  alpha=0.75, interpolation='nearest'
                  )
        # Display average membership in the image.
        for i in range(len(c_mat)):
            for j in range(len(name_list)):
                ax.text(x=j, y=i, s=round(c_mat[i, j]),
                        va='center', ha='center'
                        )

    else:  # Version 2: Main cluster counts in the table.
        ax.imshow(c_mat, cmap=plt.cm.Blues,
                  alpha=0.75, interpolation='nearest'
                  )
        # Display sample counts in the image.
        for i in range(len(c_mat)):
            for j in range(len(name_list)):
                ax.text(x=j, y=i, s=int(c_mat[i, j]),
                        va='center', ha='center'
                        )
    # Tick labels
    ax.set_xticks(range(len(name_list)))
    ax.set_xticklabels(name_list, rotation=90)
    ax.set_yticks(range(len(c_mat)))
    ax.set_yticklabels(range(len(c_mat)))
    # Axis labels
    ax.set_xlabel('Reference partition')
    ax.set_ylabel('Fuzzy clusters')
    plt.close()

    return fig, ax


def PCA_plot(FC, grouping=None, n_std=1,
             fig=None, ax=None
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

    Returns
    -------
    fig (plt.Figure)
    ax (plt.Axes)
        2D representation of the dataset using the 1st and 2nd principal
        components and confidence ellipses corresponding to each fuzzy cluster
        (or other input partition).

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
        ax = fig.add_subplot()
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
    ax.legend(ax.patches, sorted(list(set(grouping))), loc='best')
    ax.set_xlabel('1st principal component - {}% of variance'.format(
                  int(100*pca.explained_variance_ratio_[0]))
                  )
    ax.set_ylabel('2nd principal component - {}% of variance'.format(
                  int(100*pca.explained_variance_ratio_[1]))
                  )
    plt.close()

    return fig, ax


def UMAP_plot(FC, n_neighbors=75, min_dist=0.75,
              grouping=None,
              fig=None, ax=None):
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

    Returns
    -------
    fig (plt.Figure)
    ax (plt.Axes)
        A 2D representation of the dataset based on the UMAP transformation.
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
    plt.close()

    return fig, ax


def cluster_update_plot(FC, start=None, stop=None, fig=None, ax=None):
    """
    Makes a PCA plot which shows successive cluster positions.

    Arguments
    ---------
    FC (FuzzyClustering instance)
    start (int):
        which loop number to start the plot with. Solves cluster fusion issues.
    stop (int)
        which loop number to end the plot with. Solves cluster fusion issues.
    fig, ax (plt.Figure and plt.Axes instances):
        existing figure to plot in.

    Returns
    -------
    fig, ax (plt.Figure and plt.Axes)
        A plot with the successive position of the fuzzy clusters
    """

    # Run the PCA and compute the two first principal components.
    pca = PCA(n_components=2, svd_solver='full')
    princ_comp = pca.fit_transform(FC.data)
    # Plot samples in grey.
    if not fig and not ax:
        fig = plt.figure()
        ax = fig.add_subplot()
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
            ax.plot(x, y, 'x-', c=colour_set[i])
        # Highlight last cluster position.
        ax.plot(x[-1], y[-1], 'x', c=colour_set[i],
                markersize=10, markeredgewidth=2.5
                )
    plt.close()

    return fig, ax
