import numpy as np
import itertools
import lib.algorithms as al


def test_size__f(FCP_random):
    FCP_random.initiate_clusters()
    FCP_random.calculate_memberships()
    assert FCP_random._f().shape == FCP_random.memberships.shape

def test_value__f():
    FCP = al.FuzzyClusteringPoly(np.array([]), 0.5, 3)
    FCP.memberships = np.array([[1, 0, 0], [0.5, 0.2, 0.3], [1/3, 1/3, 1/3]])
    result = FCP._f()
    expected = np.array([[1, 0, 0], [5/12, 11/75, 23/100], [7/27, 7/27, 7/27]])
    assert np.isclose(result, expected).all()

def test_expectedvalue_evaluate_objective_function(unif_1D):
    FCP = FuzzyClusteringPoly(unif_1D, 0.5, 2)
    FCP.clusters.append(np.array([[3.1], [8.9]]))
    FCP.calculate_memberships()
    FCP.evaluate_objective_function()
    assert np.isclose(FCP.obj_function, 46.3283333333333333)

def test_size_calculate_memberships(FCP_random):
    FCP_random.initiate_clusters()
    FCP_random.calculate_memberships()
    expected = (FCP_random.n_samples, FCP_random.n_clusters)
    assert FCP_random.memberships.shape == expected

def test_between0and1_calculate_memberships(FCP_random):
    FCP_random.initiate_clusters()
    FCP_random.calculate_memberships()
    assert (FCP_random.memberships >= 0).all()
    assert (FCP_random.memberships <= 1).all()

def test_sumto1_calculate_memberships(FCP_random):
    FCP_random.initiate_clusters()
    FCP_random.calculate_memberships()
    result = np.sum(FCP_random.memberships, axis=1)
    assert np.isclose(result, 1).all()
