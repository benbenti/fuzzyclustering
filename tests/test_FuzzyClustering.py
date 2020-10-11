import numpy as np
import itertools
import fuzzyclustering.algorithms as al


def test_iszero__g(FC_random):
    assert FC_random._g() == 0

def test_initiate_clusters_correstsize(FC_random):
    FC_random.initiate_clusters()
    expected = (FC_random.n_samples, FC_random.data.shape[1])
    assert FC.clusters[-1].shape == expected

def test_initiate_clusters_correctrange(FC_random):
    FC_random.initiate_clusters()
    result = FC_random.clusters[-1]
    maxs = np.max(FC_random.data, axis=0)
    mins = np.min(FC_random.data, axis=0)
    for feat in range(len(maxs)):
        assert (result[:, feat] <= maxs[feat]).all()
        assert (result[:, feat] >= mins[feat]).all()

def test_calculate_memberships_correctsize(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    expected = (FC_random.n_samples, FC_random.n_clusters)
    assert FC_random.memberships.shape == expected

def test_calculate_memberships_betweenzeroandone(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    assert (FC_random.memberships >= 0).all()
    assert (FC_random.memberships <= 1).all()

def test_calculate_memberships_sumtoone(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    result = np.sum(FC_random.memberships, axis=1)
    assert np.isclose(result, 1).all()

def test_calculate_memberships_case1(case1):
    FC = al.FuzzyClustering(case1, 2, 2)
    FC.clusters.append(np.array([[0], [10]]))
    FC.calculate_memberships()
    expected = np.array([[1, 0],
                         [81/82, 1/82],
                         [16/17, 1/17],
                         [49/58, 9/58],
                         [9/13, 4/13],
                         [1/2, 1/2],
                         [4/13, 9/13],
                         [9/58, 49/58],
                         [1/17, 16/17],
                         [1/82, 81/82],
                         [0, 1]
                         ]
                        )
    assert np.isclose(FC.memberships == expected).all()

def test_calculate_memberships_case2(case2):
    FC = al.FuzzyClustering(case2, 2, 2)
    FC.clusters.append(np.array([[-3.5], [3.5]]))
    FC.calculate_memberships()
    expected = np.array([[289/298, 9/298],
                         [225/226, 1/226],
                         [169/170, 1/170],
                         [121/130, 9/130],
                         [9/130, 121/130],
                         [1/170, 169/170],
                         [1/226, 225/226],
                         [9/298, 289/298]
                         ]
                        )
    assert np.isclose(FC.memberships == expected).all()

def test_calculate_memberships_case3(case3):
    FC = al.FuzzyClustering(case3, 2, 2)
    FC.clusters.append(np.array([[1, 2], [3, 2]]))
    FC.calculate_memberships()
    expected = np.array([[13/18, 5/18],
                         [2/3, 1/3],
                         [1/2, 1/2],
                         [1/3, 2/3],
                         [5/18, 13/18],
                         [5/6, 1/6],
                         [5/6, 1/6],
                         [1/2, 1/2],
                         [5/6, 1/6],
                         [5/6, 1/6],
                         [9/10, 1/10],
                         [1, 0],
                         [1/2, 1/2],
                         [0, 1],
                         [1/10, 9/10],
                         [5/6, 1/6],
                         [5/6, 1/6],
                         [1/2, 1/2],
                         [5/6, 1/6],
                         [5/6, 1/6],
                         [13/18, 5/18],
                         [2/3, 1/3],
                         [1/2, 1/2],
                         [1/3, 2/3],
                         [5/18, 13/18]
                         ]
                        )
    assert np.isclose(FC.memberships == expected).all()

def test_calculate_memberships_case4(case4):
    FC = al.FuzzyClustering(case4, 2, 2)
    FC.clusters.append(np.array([[1, 1], [6, 6]]))
    FC.calculate_memberships()
    expected = np.array([[36/37, 1/37],
                         [61/62, 1/62],
                         [26/27, 1/27],
                         [61/62, 1/62],
                         [1, 0],
                         [41/42, 1/42],
                         [26/27, 1/27],
                         [41/42, 1/42],
                         [16/17, 1/17],
                         [1/17, 16/17],
                         [1/42, 41/42],
                         [1/27, 26/27],
                         [1/42, 41/42],
                         [0, 1],
                         [1/62, 61/62],
                         [1/27, 26/27],
                         [1/62, 61/62],
                         [1/37, 36/37]
                         ]
                        )
    assert np.isclose(FC.memberships == expected).all()

def test_calculate_memberships_case5(case5):
    FC = al.FuzzyClustering(case5, 2, 2)
    FC.clusters.append(np.array([[1, 1], [7, 7]]))
    FC.calculate_memberships()
    expected = np.array([[49/50, 1/50],
                         [85/86, 1/86],
                         [37/38, 1/38],
                         [85/86, 1/86],
                         [1, 0],
                         [61/62, 1/62],
                         [37/38, 1/38],
                         [61/62, 1/62],
                         [25/26, 1/26],
                         [1/5, 4/5],
                         [5/46, 41/46],
                         [2/27, 25/27],
                         [1/14, 13/14],
                         [1/11, 10/11],
                         [5/46, 41/46],
                         [1/37, 36/37],
                         [1/62, 61/62],
                         [1/38, 37/38],
                         [5/94, 89/94],
                         [2/27, 25/27],
                         [1/62, 61/62],
                         [0, 1],
                         [1/86, 85/86],
                         [1/26, 25/26],
                         [1/14, 13/14],
                         [1/38, 37/38],
                         [1/86, 85/86],
                         [1/65, 64/65],
                         [5/118, 113/118],
                         [1/11, 10/11],
                         [5/94, 89/94],
                         [1/26, 25/26],
                         [5/118, 113/118],
                         [1/17, 16/17],
                         ]
                        )
    assert np.isclose(FC.memberships == expected).all()

def test_calculate_memberships_case6(case6):
    FC = al.FuzzyClustering(case6, 2, 2)
    FC.clusters.append(np.array([[3, 0], [5, 0]]))
    FC.calculate_memberships()
    expected = np.array([[25/34, 9/34],
                         [17/22, 5/22],
                         [4/5, 1/5],
                         [17/22, 5/22],
                         [5/6, 1/6],
                         [9/10, 1/10],
                         [5/6, 1/6],
                         [2/3, 1/3],
                         [5/6, 1/6],
                         [1, 0],
                         [5/6, 1/6],
                         [2/3, 1/3],
                         [1/2, 1/2],
                         [1/2, 1/2],
                         [1/2, 1/2],
                         [1/3, 2/3],
                         [1/6, 5/6],
                         [0, 1],
                         [1/6, 5/6],
                         [1/3, 2/3],
                         [1/6, 5/6],
                         [1/10, 9/10],
                         [1/6, 5/6],
                         [5/22, 17/22],
                         [1/5, 4/5],
                         [5/22, 17/22],
                         [9/34, 25/34]
                         ]
                        )
    assert np.isclose(FC.memberships == expected).all()

def test__f_correctsize(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    result = FC_random._f()
    assert result.shape == FC_random.memberships.shape

def test__f_simplecase():
    dataset = np.array([])
    FC = al.FuzzyClustering(dataset, 2, 3)
    FC.memberships = np.array([[1, 0, 0], [0.5, 0.2, 0.3], [1/3, 1/3, 1/3]])
    result = FC._f()
    expected = np.array([[1, 0, 0], [0.5, 0.2, 0.3], [1/3, 1/3, 1/3]])
    assert np.isclose(result, expected).all()
    
def test_update_clusters_correctsize(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    FC_random.update_clusters()
    assert FC_random.clusters[-1].shape == FC_random.clusters[-2].shape

def test_evaluate_objective_function_correctoutput(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    FC_random.evaluate_objective_function()
    assert isinstance(FC_random.obj_function[-1], float)

def test_evaluate_objective_function_simplecase(case1):
    FC = al.FuzzyClustering(case1, 2, 2)
    FC.clusters.append(np.array([[0], [10]]))
    FC.calculate_memberships()
    FC.evaluate_objective_function()
    assert np.isclose(FC.obj_function[-1], 17.642724644851995)

def test_update_clusters_newclusterscloserfromsolution1(case2):
    FC = al.FuzzyClustering(case2, 2, 2)
    FC.initiate_clusters()
    FC.calculate_memberships()
    FC.evaluate_objective_function()
    FC.update_clusters()
    FC.calculate_memberships()
    FC.evaluate_objective_function()
    assert FC.obj_function[-1] <= FC.obj_function[-2]

def test_update_clusters_newclusterscloserfromsolution2(case4):
    FC = al.FuzzyClustering(case4, 2, 2)
    FC.initiate_clusters()
    FC.calculate_memberships()
    FC.evaluate_objective_function()
    FC.update_clusters()
    FC.calculate_memberships()
    FC.evaluate_objective_function()
    assert FC.obj_function[-1] <= FC.obj_function[-2]

def test_update_clusters_newclusterscloserfromsolution3(case5):
    FC = al.FuzzyClustering(case5, 2, 2)
    FC.initiate_clusters()
    FC.calculate_memberships()
    FC.evaluate_objective_function()
    FC.update_clusters()
    FC.calculate_memberships()
    FC.evaluate_objective_function()
    assert FC.obj_function[-1] <= FC.obj_function[-2]

def test_update_clusters_newclusterscloserfromsolution4(case6):
    FC = al.FuzzyClustering(case6, 2, 2)
    FC.initiate_clusters()
    FC.calculate_memberships()
    FC.evaluate_objective_function()
    FC.update_clusters()
    FC.calculate_memberships()
    FC.evaluate_objective_function()
    assert FC.obj_function[-1] <= FC.obj_function[-2]

def test_VIdso_correctsize(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    FC_random.VIdso()
    assert len(FC_random.cluster_quality) == 3

def test_VIdso_simplecase(case1, case2):
    FC1 = al.FuzzyClustering(case1, 2, 2)
    FC1.clusters.append(np.array([[1.5], [8.5]]))
    FC1.calculate_memberships()
    FC1.VIdso()
    FC2 = al.FuzzyClustering(case2, 2, 2)
    FC2.clusters.append(np.array([[-3.5], [3.5]]))
    FC2.calculate_memberships()
    FC2.VIdso()
    assert FC1.cluster_quality[0] <= FC2.cluster_quality[0]  # dispersion
    assert FC1.cluster_quality[1] <= FC2.cluster_quality[1]  # separation
    assert FC2.cluster_quality[2] <= FC1.cluster_quality[2]  # overlap

def test_intrainter_silhouette_betweenoneandminusone(FC_random):
    FC_random.initiate_clusters()
    FC_random.calculate_memberships()
    FC_random.intrainter_silhouette()
    assert (-1 <= FC_random.cluster_quality <= 1).all()

def test_intrainter_silhouette_simplecase(case5):
    FC = al.FuzzyClustering(case5, 2, 2)
    FC.clusters.append(np.array([[1, 1], [7, 7]]))
    FC.intrainter_silhouette()
    assert (FC.cluster_quality[1, 2, 5] == FC.cluster_quality[3, 6, 7]).all()
    assert(FC.cluster_quality[9] <= FC.cluster_quality[15] <= FC.cluster_quality[21])
