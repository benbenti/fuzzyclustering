# import numpy as np
# import fuzzyclustering.algorithms as al
# import math

# def test__f_sizematch(FCP_sample):
    # FCP_sample.initiate_clusters()
    # FCP_sample.calculate_memberships()
    # result = FCP_sample._f()
    # assert result.shape == FCP_sample.memberships.shape

# def test__f_simple_case():
    # dataset = np.array([])
    # FCP = al.FuzzyClusteringPoly(dataset, 0.5, 2)
    # FCP.memberships = np.array([[1, 0, 0], [0.5, 0.2, 0.3], [1/3, 1/3, 1/3]])
    # result = FCP._f()
    # cmp = np.isclose(result, np.array([[1, 0, 0], [5/12, 11/75, 23/100], [7/27, 7/27, 7/27]]))
    # assert cmp.all()

# def test_calculate_memberships_sizematch(FCP_sample):
    # FCP_sample.initiate_clusters()
    # FCP_sample.calculate_memberships()
    # assert FCP_sample.memberships.shape == (FCP_sample.n_samples, FCP_sample.n_clusters)

# def test_calculate_memberships_betweenzeroandone(FCP_sample):
    # FCP_sample.initiate_clusters()
    # FCP_sample.calculate_memberships()
    # assert (FCP_sample.memberships >= 0).all()
    # assert (FCP_sample.memberships <= 1).all()

# def test_calculate_memberships_sumtoone(FCP_sample):
    # FCP_sample.initiate_clusters()
    # FCP_sample.calculate_memberships()
    # result = np.sum(FCP_sample.memberships, axis=1)
    # assert np.isclose(result, 1).all()

# def test_calculate_memberships_simplecase():
    # dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # ct = np.array([0, 10])
    # nc = 2
    # p = 0.25
    # FCP = al.FuzzyClusteringPoly(dataset, p, nc)
    # FCP.clusters.append(ct)
    # FCP.calculate_memberships()
    # mb = FCP.memberships
    # # Check if membership scores match distance ratios.
    # assert (mb[:6,0] == mb[-1:-7:-1, 1]).all()

# def test_calculate_memberships_clusteronsample():
    # # Raises warnings because division by zero.
    # dataset = np.array([[0, 0],
                        # [0, 1],
                        # [1, 0]
                        # ]
                       # )
    # nc = 2
    # p = 0.5
    # FCP = al.FuzzyClusteringPoly(dataset, p, nc)
    # FCP.clusters = [np.array([[0, 0],[6, 6]])]
    # FCP.calculate_memberships()
    # mb = FCP.memberships
    # assert (mb[0] == [1, 0]).all()

# def test_evaluate_objective_function():
    # dataset = np.array([[0], [5], [10]])
    # nc = 2
    # p = 0.5
    # FCP = al.FuzzyClusteringPoly(dataset, p, nc)
    # FCP.clusters.append(np.array([[0], [10]]))
    # FCP.calculate_memberships()
    # FCP.evaluate_objective_function()
    # assert math.isclose(FCP.obj_function[-1], 125/6)