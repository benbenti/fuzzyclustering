# import numpy as np
# import fuzzyclustering.algorithms as al
# import math

# def test__f_isidentity(FCRS_sample):
    # FCRS_sample.initiate_clusters()
    # FCRS_sample.calculate_memberships()
    # result = FCRS_sample._f()
    # assert (result == FCRS_sample.memberships).all()

# def test__g_sizematch(FCRS_sample):
    # FCRS_sample.initiate_clusters()
    # FCRS_sample.calculate_memberships()
    # result = FCRS_sample._g()
    # assert result.shape == FCRS_sample.memberships.shape

# def test__g_simplecase():
    # dataset = np.array([])
    # FCRS = al.FuzzyClusteringRegulSh(dataset, 5, 2)
    # FCRS.memberships = np.array([[1, 0], [0.5, 0.5], [1/3, 2/3]])
    # result = FCRS._g()
    # expected = np.array([[0, 0], [-2.5*np.log(0.5), -2.5*np.log(0.5)], [-(5/3)*np.log(1/3), -(10/3)*np.log(2/3)]])
    # cmp = np.isclose(result, expected)
    # assert cmp.all()

# def test_calculate_memberships_sizematch(FCRS_sample):
    # FCRS_sample.initiate_clusters()
    # FCRS_sample.calculate_memberships()
    # assert FCRS_sample.memberships.shape == (FCRS_sample.n_samples, FCRS_sample.n_clusters)

# def test_calculate_memberships_betweenzeroandone(FCRS_sample):
    # FCRS_sample.initiate_clusters()
    # FCRS_sample.calculate_memberships()
    # assert (FCRS_sample.memberships >= 0).all()
    # assert (FCRS_sample.memberships <= 1).all()

# def test_calculate_memberships_sumtoone(FCRS_sample):
    # FCRS_sample.initiate_clusters()
    # FCRS_sample.calculate_memberships()
    # result = np.sum(FCRS_sample.memberships, axis=1)
    # assert np.isclose(result, 1).all()

# def test_calculate_memberships_simple_case():
    # dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # ct = np.array([0, 10])
    # nc = 2
    # p = 5
    # FCRS = al.FuzzyClusteringRegulSh(dataset, p, nc)
    # FCRS.clusters.append(ct)
    # FCRS.calculate_memberships()
    # mb = FCRS.memberships
    # # Check if membership scores match distance ratios.
    # assert (mb[:6,0] == mb[-1:-7:-1, 1]).all()

# def test_evaluate_objective_function():
    # dataset = np.array([[0], [5], [10]])
    # nc = 2
    # p = 5
    # FCRS = al.FuzzyClusteringRegulSh(dataset, p, nc)
    # FCRS.clusters.append(np.array([[0], [10]]))
    # FCRS.calculate_memberships()
    # FCRS.evaluate_objective_function()
    # expected = 25 - 5 * np.log(0.5)
    # assert math.isclose(FCRS.obj_function[-1], expected, abs_tol=0.00001)
