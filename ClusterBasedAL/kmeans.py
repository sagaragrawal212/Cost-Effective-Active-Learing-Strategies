import numpy as np
from sklearn.cluster import KMeans

class KMeansClusterer:
    def __init__(self, data, k=10):
        self.data = data
        self.k = k
        self.kmeans = KMeans(n_clusters=k)
        self.used_indices = []

    def fit(self):
        self.kmeans.fit(self.data)

    def get_top_indices(self, num_top_indices=100):
        clustered_indices = []
        resulting_indices = []
        for cluster_id in range(self.k):
            top_indices = []
            if cluster_id not in clustered_indices:
                cluster_distances = self.kmeans.transform(self.data)[:, cluster_id]
                cluster_indices = np.argsort(cluster_distances)

                for idx in cluster_indices:
                    if idx not in self.used_indices:
                        top_indices.append(idx)
                        self.used_indices.append(idx)

                    if len(top_indices) >= num_top_indices:
                      resulting_indices.extend(top_indices)
                      break

                clustered_indices.append(cluster_id)

        return resulting_indices
