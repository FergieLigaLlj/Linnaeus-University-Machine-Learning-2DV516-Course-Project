###    Use the class to define the k-means algorithm first
import numpy as np
class kmeans:
    # initialization
    def __init__(self,k=2, tolerance = 0.001 , maximum_iteration = 500 ):
        self.k = k
        self.maximum_iteration = maximum_iteration
        self.tolerance = tolerance
    # euclidean distance
    def euclidean_distance(self,point_1,point_2):
        return np.linalg.norm(point_1-point_2,axis=0)
    # assign centroids
    def fitted_centroids(self,X):
        self.centroids = {}
        choice = np.random.choice(range(0,len(X)),self.k)
        for i in range(self.k):
            self.centroids[i] = X[choice[i]]
        for i in range(self.maximum_iteration):
            self.classes = {}
            for j in range(self.k):
                self.classes[j] = []
            for x in X:
                distances = []
                for index in self.centroids:
                    distances.append(self.euclidean_distance(x,self.centroids[index]))
                cluster_index = distances.index(min(distances))
                self.classes[cluster_index].append(x)
            previous = dict(self.centroids)
            for cluster_index in self.classes:
                self.centroids[cluster_index] = np.average(self.classes[cluster_index],axis = 0)
            isoptimal = True
            for centroid in self.centroids:
                initial_centroid = previous[centroid]
                current = self.centroids[centroid]
                if np.sum((current - initial_centroid)/initial_centroid * 100.0) > self.tolerance:
                    isoptimal = False
            if isoptimal:
                break
    def np_change(self,X):
        dictionary = []
        for cluster_index in X:
            for features in X[cluster_index]:
                dictionary.append((features[0],features[1],int(cluster_index)))
        dictionary = np.array(dictionary)
        return dictionary
###    Finish line