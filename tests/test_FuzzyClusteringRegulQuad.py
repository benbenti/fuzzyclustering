import numpy as np
import itertools
import lib.algorithms as al


def test_isidentity__f(FCRQ_random):
    FCRQ_random.initiate_clusters()
    FCRQ_random.calculate_memberships()
    assert (FCRQ_random._f() == FCRQ_random.memberships).all()

def test_size__g(FCRQ_random):
    FCRQ_random.initiate_clusters()
    FCRQ_random.calculate_memberships()
    assert FCRQ_random._g().shape == FCRQ_random.memberships.shape

def test_expectedvalue__g():
    FCRQ = al.FuzzyClusteringRegulQuad(np.array([]), 5, 2)
    FCRQ.memberships = np.array([[1, 0], [0.5, 0.5], [1/3, 2/3]])
    result = FCRQ._g()
    expected = np.array([[5, 0], [5*0.25, 5*0.25], [5*1/9, 5*4/9]])
    assert (np.isclose(result, expected)).all()

def test_expectedvalue_evaluate_objective_function(unif_1D):
    FCRQ = FuzzyClusteringRegulQuad(unif_1D, 5, 2)
    FCRQ.clusters.append(np.array([[3], [9]]))
    FCRQ.calculate_memberships()
    FCRQ.evaluate_objective_function()
    assert np.isclose(FCRQ.obj_function, 317.3660240542267)

def test_size_calculate_memberships(FCRQ_random):
    FCRQ_random.initiate_clusters()
    FCRQ_random.calculate_memberships()
    expected = (FCRQ_random.n_samples, FCRQ_random.n_clusters)
    assert FCRQ_random.memberships.shape == expected

def test_between0and1_calculate_memberships(FCRQ_random):
    FCRQ_random.initiate_clusters()
    FCRQ_random.calculate_memberships()
    assert (FCRQ_random.memberships >= 0).all()
    assert (FCRQ_random.memberships <= 1).all()

def test_sumto1_calculate_memberships(FCRQ_random):
    FCRQ_random.initiate_clusters()
    FCRQ_random.calculate_memberships()
    result = np.sum(FCRQ_random.memberships, axis=1)
    assert np.isclose(result, 1).all()
