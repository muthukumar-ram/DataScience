# -*- coding: utf-8 -*-
"""


@author: inmkumar10

Kmeans algorithm implemented. Image is available in data folder
"""

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy import misc

bean = misc.imread('bean.jpg')
type(bean)
x,y,z = bean.shape
bean2d = bean.reshape(x*y,z)
bean2d.shape
kmeans = KMeans(n_clusters=3)
kmeans.fit(bean2d)
cluster_center = kmeans.cluster_centers_
cluster_labels = kmeans.labels_
plt.figure(figsize = (15,8))
clus = cluster_center[cluster_labels]
clus = clus.reshape(x,y,z)
plt.imshow(clus.astype('f'))

