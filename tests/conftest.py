import pytest
import random
import numpy as np
from numpy.random import rand
import lib.algorithms as al

@pytest.fixture(scope="session")
def unif_1D():
    """
    Test case: one dimension, samples evenly distributed.
    """
    data = np.array([[0], [1], [2], [3], [4], [5], [6],
                     [7], [8], [9], [10], [11], [12]
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

# @pytest.fixture
# def FCP_random(rng, dataset, nc):
#     p = rng.random()
#     return al.FuzzyClusteringPoly(dataset, p, nc)

# @pytest.fixture
# def FCRS_random(rng, dataset, nc):
#     p = rng.random() * 5
#     return al.FuzzyClusteringRegulSh(dataset, p, nc)

# @pytest.fixture
# def FCRQ_random(rng, dataset, nc):
#     p = rng.random() * 5
#     return al.FuzzyClusteringRegulQuad(dataset, p, nc)
