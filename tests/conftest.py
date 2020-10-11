import pytest
import random
import numpy as np
from numpy.random import rand
import fuzzyclustering.algorithms as al

@pytest.fixture(scope="session")
def case1():
    """
    Simple case for testing: one dimension, samples evenly distributed.
    """
    data = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    return data

@pytest.fixture(scope="session")
def case2():
    """
    Simple case for testing: one dimension, two distinct clusters.
    """
    data = np.array([[-5],[-4],[-3],[-2], [2],[3],[4],[5]])
    return data

@pytest.fixture(scope="session")
def case3():
    """
    Simple case for testing: two dimensions, samples evenly
    distributed.
    """
    data = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                     [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
                     [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
                     [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
                     [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def case4():
    """
    Simple case for testing: two dimensions, two distinct, identical
    in shape, clusters.
    """
    data = np.array([[0, 0], [0, 1], [0, 2],
                     [1, 0], [1, 1], [1, 2],
                     [2, 0], [2, 1], [2, 2],
                     [5, 5], [5, 6], [5, 7],
                     [6, 5], [6, 6], [6, 7],
                     [7, 5], [7, 6], [7, 7]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def case5():
    """
    Simple case for testing: two dimensions, two distinct, different in
    size clusters.
    """
    data = np.array([[0, 0], [0, 1], [0, 2],
                     [1, 0], [1, 1], [1, 2],
                     [2, 0], [2, 1], [2, 2],
                     [5, 5], [5, 6], [5, 7], [5, 8], [5, 9],
                     [6, 5], [6, 6], [6, 7], [6, 8], [6, 9],
                     [7, 5], [7, 6], [7, 7], [6, 8], [6, 9],
                     [8, 5], [8, 6], [8, 7], [8, 8], [8, 9],
                     [9, 5], [9, 6], [9, 7], [9, 8], [9, 9]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def case6():
    """
    Simple case for testing: two dimension, two overlapping, identical
    clusters.
    """
    data = np.array([[0, 0],
                     [1, -1], [1, 0], [1, 1],
                     [2, -1], [2, 0], [2, 1],
                     [3, -2], [3, -1], [3, 0], [3, 1], [3, 2],
                     [4, -1], [4, 0], [4, 1],
                     [5, -2], [5, -1], [5, 0], [5, 1], [5, 2],
                     [6, -1], [6, 0], [6, 1],
                     [7, -1], [7, 0], [7, 1],
                     [8, 0]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def rng():
    return random.Random()

@pytest.fixture
def dataset(rng):
    n_samples = rng.randint(100, 1000)
    n_features = rng.randint(10, 100)
    feature_range = rng.randint(1, 10)
    return (rand(n_samples, n_features) - 1/2) * feature_range

@pytest.fixture
def nc(rng):
    return rng.randint(2, 50)

@pytest.fixture
def FC_random(rng, dataset, nc):
    p = 1 + rng.random() * 2
    return al.FuzzyClustering(dataset, p, nc)

@pytest.fixture
def FCP_random(rng, dataset, nc):
    p = rng.random()
    return al.FuzzyClusteringPoly(dataset, p, nc)

@pytest.fixture
def FCRS_random(rng, dataset, nc):
    p = rng.random() * 10
    return al.FuzzyClusteringRegulSh(dataset, p, nc)

@pytest.fixture
def FCRQ_random(rng, dataset, nc):
    p = rng.random() * 10
    return al.FuzzyClusteringRegulQuad(dataset, p, nc)
