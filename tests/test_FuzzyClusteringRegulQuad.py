# import numpy as np
# import fuzzyclustering.algorithms as al
# import math

# def test__f_isidentity(FCRQ_sample):
    # FCRQ_sample.initiate_clusters()
    # FCRQ_sample.calculate_memberships()
    # result = FCRQ_sample._f()
    # assert (result == FCRQ_sample.memberships).all()

# def test__g_sizematch(FCRQ_sample):
    # FCRQ_sample.initiate_clusters()
    # FCRQ_sample.calculate_memberships()
    # result = FCRQ_sample._g()
    # assert result.shape == FCRQ_sample.memberships.shape

# def test__g_simplecase():
    # dataset = np.array([])
    # FCRQ = al.FuzzyClusteringRegulQuad(dataset, 5, 2)
    # FCRQ.memberships = np.array([[1, 0], [0.5, 0.5], [1/3, 2/3]])
    # result = FCRQ._g()
    # expected = np.array([[5, 0], [1.25, 1.25], [5/9, 20/9]])
    # cmp = np.isclose(result, expected)
    # assert cmp.all()

# def test_calculate_memberships_sizematch(FCRQ_sample):
    # FCRQ_sample.initiate_clusters()
    # FCRQ_sample.calculate_memberships()
    # assert FCRQ_sample.memberships.shape == (FCRQ_sample.n_samples, FCRQ_sample.n_clusters)

# def test_calculate_memberships_betweenzeroandone(FCRQ_sample):
    # FCRQ_sample.initiate_clusters()
    # FCRQ_sample.calculate_memberships()
    # assert (FCRQ_sample.memberships >= 0).all()
    # assert (FCRQ_sample.memberships <= 1).all()

# def test_calculate_memberships_sumtoone(FCRQ_sample):
    # FCRQ_sample.initiate_clusters()
    # FCRQ_sample.calculate_memberships()
    # result = np.sum(FCRQ_sample.memberships, axis=1)
    # assert np.isclose(result, 1).all()

# def test_calculate_memberships_simple_case():
    # dataset = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
    # ct = np.array([[0], [10]])
    # nc = 2
    # p = 5
    # FCRQ = al.FuzzyClusteringRegulQuad(dataset, p, nc)
    # FCRQ.clusters.append(ct)
    # FCRQ.calculate_memberships()
    # mb = FCRQ.memberships
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
    # assert FCRS.obj_function[-1] == 37.5
