import numpy as np
import itertools
import lib.algorithms as al


def test_size__f(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    assert FC_random._f().shape == FC_random.memberships.shape

def test_value__f():
    FC = al.FuzzyClustering(np.array([]), 2, 3)
    FC.memberships = np.array([[1, 0, 0], [0.5, 0.2, 0.3], [1/3, 1/3, 1/3]])
    result = FC._f()
    expected = np.array([[1, 0, 0], [0.25, 0.04, 0.09], [1/9, 1/9, 1/9]])
    assert np.isclose(result, expected).all()

def test_iszero__g(FC_random):
    assert FC_random._g() == 0

def test_size_initiate_clusters(FC_random):
    FC_random.initiate_clusters()
    expected = (FC_random.n_clusters, FC_random.data.shape[1])
    assert FC_random.clusters[-1].shape == expected

def test_range_initiate_clusters(FC_random):
    FC_random.initiate_clusters()
    result = FC_random.clusters[-1]
    maxs = np.max(FC_random.data, axis=0)
    mins = np.min(FC_random.data, axis=0)
    for feat in range(FC_random.data.shape[1]):
        assert (mins[feat] <= result[:, feat]).all()
        assert (result[:, feat] <= maxs[feat]).all()

def test_size_calculate_memberships(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    expected = (FC_random.n_samples, FC_random.n_clusters)
    assert FC_random.memberships.shape == expected

def test_between0and1_calculate_memberships(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    assert (FC_random.memberships >= 0).all()
    assert (FC_random.memberships <= 1).all()

def test_sumto1_calculate_memberships(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    result = np.sum(FC_random.memberships, axis=1)
    assert np.isclose(result, 1).all()

def test_size_update_clusters(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    FC_random.update_clusters()
    assert FC_random.clusters[-1].shape == FC_random.clusters[-2].shape

def test_range_update_clusters(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    FC_random.update_clusters()
    ct = FC_random.clusters[-1]
    maxs = np.max(FC_random.data, axis=0)
    mins = np.min(FC_random.data, axis=0)
    for feat in range(FC_random.data.shape[1]):
        assert (mins[feat] <= ct[:, feat]).all()
        assert (ct[:, feat] <= maxs[feat]).all()

def test_expectedvalue_evaluate_objective_function(unif_1D):
    FC = FuzzyClustering(unif_1D, 2, 2)
    FC.clusters.append(np.array([[3], [9]]))
    FC.calculate_memberships()
    FC.evaluate_objective_function()
    assert np.isclose(FC.obj_function, 38.5124886877828056)

def test_size_VIdso(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    FC_random.VIdso()
    assert len(FC_random.cluster_quality) == 3

def test_size_intrainter_silhouette(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    FC_random.intrainter_silhouette()
    assert len(FC_random.cluster_quality) == FC_random.n_samples

def test_betweenminus1and1_intraintersilhouette(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    FC_random.intrainter_silhouette()
    assert (FC_random.cluster_quality <= 1).all()
    assert (FC_random.cluster_quality >= -1).all()
