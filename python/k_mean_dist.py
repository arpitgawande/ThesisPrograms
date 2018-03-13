
# coding: utf-8

# In[ ]:

from scipy.spatial.distance import euclidean
#Calculate distance of feature vector point from the its cluster center
def k_mean_dist(feature_vector, label, cluster_centers):
    distances = []
    for i, d in enumerate(feature_vector):
        center = cluster_centers[label[i]]
        distance = euclidean(d,center)
        #distance = np.linalg.norm(d - center)
        distances.append(distance)
    return distances


# In[ ]:

distances = k_mean_dist(X_trans, clusters, centroids)
#Getting the labels/clusters and distances of each IP from centroid
distance_df = pd.DataFrame({'distance': distances})

