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
def sparse_close_1D():
    """
    Test case: one dimension, two sparse clusters, close from each other.
    """
    data = np.array([[0], [1], [2], [3], [4], [5],
                     [7], [8], [9], [10], [11], [12]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def dense_close_1D():
    """
    Test case: one dimension, two dense clusters, close from each other.
    """
    data = np.array([[0], [2], [2+1/3], [2+2/3], [3], [5],
                     [7], [9], [9+1/3], [9+2/3], [10], [12]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def sparse_far_1D():
    """
    Test case: one dimension, two sparse clusters, far from each other.
    """
    data = np.array([[0], [1], [2], [3], [4], [5],
                     [10], [11], [12], [13], [14], [15]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def dense_far_1D():
    """
    Test case: one dimension, two dense clusters, far from each other.
    """
    data = np.array([[0], [2], [2+1/3], [2+2/3], [3], [5],
                     [10], [12], [12+1/3], [12+2/3], [13], [15]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def complex_1D():
    """
    Test case: one dimension, four clusters with different density.
    """
    data = np.array([[0], [1],
                     [4], [4+1/3], [4+2/3], [5],
                     [7], [7+1/3], [7+2/3], [8],
                     [11], [12]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def unif_2D():
    data = np.array([[-3, 0],
                     [-2, -1], [-2, 0], [-2, 1],
                     [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
                     [0, -3], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [0, 3],
                     [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                     [2, -1], [2, 0], [2, 1],
                     [3, 0]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def sparse_close_2D():
    """
    Test case: two dimension, two sparse clusters, close from each other.
    """
    data = np.array([[0, -1], [0, 1],
                     [1, -2], [1, -1], [1, 1], [1, 2],
                     [2, -3], [2, -2], [2, -1], [2, 1], [2, 2], [2, 3],
                     [3, -3], [3, -2], [3, -1], [3, 1], [3, 2], [3, 3],
                     [4, -2], [4, -1], [4, 1], [4, 2],
                     [5, -1], [5, 1],

                     [7, -1], [7, 1],
                     [8, -2], [8, -1], [8, 1], [8, 2],
                     [9, -3], [9, -2], [9, -1], [9, 1], [9, 2], [9, 3],
                     [11, -3], [11, -2], [11, -1], [11, 1], [11, 2], [11, 3],
                     [10, -2], [10, -1], [10, 1], [10, 2],
                     [11, -1], [11, 1],
                     [12, 2], [12, 2]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def sparse_far_2D():
    """
    Test case: two dimension, two sparse clusters, far from each other.
    """
    data = np.array([[0, -1], [0, 1],
                     [1, -2], [1, -1], [1, 1], [1, 2],
                     [2, -3], [2, -2], [2, -1], [2, 1], [2, 2], [2, 3],
                     [3, -3], [3, -2], [3, -1], [3, 1], [3, 2], [3, 3],
                     [4, -2], [4, -1], [4, 1], [4, 2],
                     [5, -1], [5, 1],

                     [10, -1], [10, 1],
                     [11, -2], [11, -1], [11, 1], [11, 2],
                     [12, -3], [12, -2], [12, -1], [12, 1], [12, 2], [12, 3],
                     [13, -3], [13, -2], [13, -1], [13, 1], [13, 2], [13, 3],
                     [14, -2], [14, -1], [14, 1], [14, 2],
                     [15, -1], [15, 5],
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def dense_close_2D():
    """
    Test case: two dimension, two dense clusters, close from each other.
    """
    data = np.array([[0, -1], [0, 1],
                     [1+1/2, -1/3], [1+1/2, 1/3]
                     [2, -3], [2, -1], [2, 1], [2, 3],
                     [2+1/3, -3/2], [2+1/3, -1/3], [2+1/3, 1/3], [2+1/3, 3/2],
                     [2+2/3, -3/2], [2+2/3, -1/3], [2+2/3, 1/3], [2+2/3, 3/2],
                     [3, -3], [3, -1], [3, 1], [3, 3],
                     [3+1/2, -1/3], [3+1/2, 1/3],
                     [5, -1], [5, 1],

                     [7, -1], [7, 1],
                     [8+1/2, -1/3], [8+1/2, 1/3]
                     [9, -3], [9, -1], [9, 1], [9, 3],
                     [9+1/3, -3/2], [9+1/3, -1/3], [9+1/3, 1/3], [9+1/3, 3/2],
                     [9+2/3, -3/2], [9+2/3, -1/3], [9+2/3, 1/3], [9+2/3, 3/2],
                     [10, -3], [10, -1], [10, 1], [10, 3],
                     [10+1/2, -1/3], [10+1/2, 1/3],
                     [12, -1], [12, 1]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def dense_far_2D():
    """
    Test case: two dimension, two dense clusters, far from each other.
    """
    data = np.array([[0, -1], [0, 1],
                     [1+1/2, -1/3], [1+1/2, 1/3]
                     [2, -3], [2, -1], [2, 1], [2, 3],
                     [2+1/3, -3/2], [2+1/3, -1/3], [2+1/3, 1/3], [2+1/3, 3/2],
                     [2+2/3, -3/2], [2+2/3, -1/3], [2+2/3, 1/3], [2+2/3, 3/2],
                     [3, -3], [3, -1], [3, 1], [3, 3],
                     [3+1/2, -1/3], [3+1/2, 1/3],
                     [5, -1], [5, 1],

                     [10, -1], [10, 1],
                     [11+1/2, -1/3], [11+1/2, 1/3]
                     [12, -3], [12, -1], [12, 1], [12, 3],
                     [12+1/3, -3/2], [12+1/3, -1/3], [12+1/3, 1/3], [12+1/3, 3/2],
                     [12+2/3, -3/2], [12+2/3, -1/3], [12+2/3, 1/3], [12+2/3, 3/2],
                     [13, -3], [13, -1], [13, 1], [13, 3],
                     [13+1/2, -1/3], [13+1/2, 1/3],
                     [15, -1], [14, 1]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def diffsize_2D():
    """
    Test case: two dimensions, two different size clusters.
    """
    data = np.array([[0, -2], [0, -1], [0, 1], [0, 2],
                     [1, -2], [1, -1], [1, 1], [1, 2],
                     [2, -2], [2, -1], [2, 1], [2, 2],
                     [3, -2], [3, -1], [3, 1], [3, 2],
                     [5, -3], [5, -2], [5, -1], [5, 1], [5, 2], [5, 3],
                     [6, -3], [6, -2], [6, -1], [6, 1], [6, 2], [6, 3],
                     [7, -3], [7, -2], [7, -1], [7, 1], [7, 2], [7, 3],
                     [8, -3], [8, -2], [8, -1], [8, 1], [8, 2], [8, 3],
                     [9, -3], [9, -2], [9, -1], [9, 1], [9, 2], [9, 3],
                     [10, -3], [10, -2], [10, -1], [10, 1], [10, 2], [10, 3]
                     ]
                    )
    return data

@pytest.fixture(scope="session")
def complex_2D():
    """
    Test case: two dimensions, six clusters of different size and densities.
    """
    data = np.array([[0, 7],
                     [1, 6], [1, 7], [1, 8],
                     [2, 5], [2, 6], [2, 7],
                     [3, 6], [3, 7],

                     [0, 10], [0, 11],
                     [1, 9], [1, 10], [1, 11],
                     [2, 8], [2, 9], [2, 10], [2, 11],
                     [3, 10], [3, 11],

                     [7, 3],
                     [8, 2], [8, 3], [8, 4],
                     [9, 2], [9, 3], [9, 4],
                     [10, 2], [10, 3], [10 ,4],
                     [11, 3],

                     [13, 3],
                     [14, 2], [14, 3], [14, 4],
                     [15, 3],
                     [16, 2], [16, 3], [16, 4],
                     [17, 3],

                     [9, 8], [9, 10],
                     [11, 7], [11, 9], [11, 11],
                     [13, 8], [13, 11],
                     [13.5, 11.5],
                     [14, 11],

                     [15, 12],
                     [16, 12], [16, 13], [16, 14],
                     [18, 13], [18, 15],

                     [4, 2],

                     [5, 17]
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
    p = rng.random() * 5
    return al.FuzzyClusteringRegulSh(dataset, p, nc)

@pytest.fixture
def FCRQ_random(rng, dataset, nc):
    p = rng.random() * 5
    return al.FuzzyClusteringRegulQuad(dataset, p, nc)
