from tools import load_data, select_Minpts, chose_central_rangebins
import numpy as np
from math import pow
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

rang_bins = [i for i in range(60)]
doppler_bins = [i for i in range(24)] + [i for i in range(27, 50)]
data_2d, label = load_data(data_path="/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-23_kitchen/46-d7")
data_central = []
for d2 in data_2d:
	left, right = chose_central_rangebins(d2, DISCARD_BINS=[24,25,26])
	fft_gt = np.sum(d2[:, left:right], axis=1)
	fft_gt = fft_gt[doppler_bins]
	data_central.append(fft_gt)
data_central = np.array(data_central)
# data_central = StandardScaler().fit_transform(data_central)
# data = np.array([d.flatten() for d in data_central])
# k = pow(2, data.shape[1]) - 1

# # find best k, eps, min_pts
# k = 10
# _, nn_dist = select_Minpts(data_central, k)
#
# for k in range(100, 1000, 100):
# 	k_dist = nn_dist[:, k]
# 	k_dist.sort()
# 	plt.plot(np.arange(k_dist.shape[0]), k_dist[::-1])
# 	plt.title("k={}".format(k))
# 	plt.show()

eps = 0.65
Minpts = 301
DBscan_model = DBSCAN(eps=eps, min_samples=Minpts).fit(data_central)
# core_samples = cluster_DBscan.core_sample_indices_
pre = DBscan_model.labels_

if -1 in pre:
	pre_ = pre + 1
	print(np.bincount(pre_))

pre = DBscan_model.labels_
n_clusters_ = len(set(pre)) - (1 if -1 in pre else 0)
# 模型评估
print('估计的聚类个数为: %d' % n_clusters_)
print("同质性: %0.3f" % metrics.homogeneity_score(label, pre))  # 每个群集只包含单个类的成员。
print("完整性: %0.3f" % metrics.completeness_score(label, pre))  # 给定类的所有成员都分配给同一个群集。
print("V-measure: %0.3f" % metrics.v_measure_score(label, pre))  # 同质性和完整性的调和平均
print("调整兰德指数: %0.3f" % metrics.adjusted_rand_score(label, pre))
print("调整互信息: %0.3f" % metrics.adjusted_mutual_info_score(label, pre))
print("轮廓系数: %0.3f" % metrics.silhouette_score(data_central, pre))


