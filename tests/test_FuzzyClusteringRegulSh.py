import numpy as np
import itertools
import lib.algorithms as al


def test_isidentity__f(FCRS_random):
    FCRS_random.initiate_clusters()
    FCRS_random.calculate_memberships()
    assert (FCRS_random._f() == FCRS_random.memberships).all()

def test_size__g(FCRS_random):
    FCRS_random.initiate_clusters()
    FCRS_random.calculate_memberships()
    assert FCRS_random._g().shape == FCRS_random.memberships.shape

def test_expectedvalue__g():
    FCRS = al.FuzzyClusteringRegulSh(np.array([]), 5, 2)
    FCRS.memberships = np.array([[0.9, 0.1], [0.5, 0.5], [1/3, 2/3]])
    result = FCRS._g()
    expected = np.array([[4.5*np.log(0.9), 0.5*np.log(0.1)], [2.5*np.log(0.5), 2.5*np.log(0.5)], [(5/3)*np.log(1/3), (10/3)*np.log(2/3)]])
    assert (np.isclose(result, expected)).all()

def test_expectedvalue_evaluate_objective_function(unif_1D):
    FCRS = FuzzyClusteringRegulSh(unif_1D, 5, 2)
    FCRS.clusters.append(np.array([[3], [9]]))
    FCRS.calculate_memberships()
    FCRS.evaluate_objective_function()
    assert np.isclose(FCRS.obj_function, 42.57573454949287)

def test_size_calculate_memberships(FCRS_random):
    FCRS_random.initiate_clusters()
    FCRS_random.calculate_memberships()
    expected = (FCRS_random.n_samples, FCRS_random.n_clusters)
    assert FCRS_random.memberships.shape == expected

def test_between0and1_calculate_memberships(FCRS_random):
    FCRS_random.initiate_clusters()
    FCRS_random.calculate_memberships()
    assert (FCRS_random.memberships >= 0).all()
    assert (FCRS_random.memberships <= 1).all()

def test_sumto1_calculate_memberships(FCRS_random):
    FCRS_random.initiate_clusters()
    FCRS_random.calculate_memberships()
    result = np.sum(FCRS_random.memberships, axis=1)
    assert np.isclose(result, 1).all()
