from scipy.io import loadmat
import math
import numpy as np
import matplotlib.pyplot as plt

def dist(x, y):
    distance = math.sqrt(math.pow(x[0] - y[0], 2) + math.pow(x[1] - y[1], 2))
    return distance

def getNeighbor(Datasets, point, eps):
    neighbors = []
    for i in range(len(Datasets)):
        if dist(Datasets[point], Datasets[i]) < eps:
            neighbors.append(i)
    return neighbors

def DBSCAN(Datasets, eps, Minpoints):
    #initial all labels are 0 (0 means point has not been considered yet)
    labels = [0] * len(Datasets)
    #value is the ID of current cluster
    cluster = 0

    for point in range(len(Datasets)):
        if not (labels[point] == 0):
            continue
        #find all of point's neighboring points
        NeighborPoints = getNeighbor(Datasets, point, eps)
        if len(NeighborPoints) < Minpoints:
            #if the number is below Minpoints, this point is noise(-1)
            labels[point] = -1
        else:
            #otherwise use this point as the seed for a new cluster
            cluster += 1
            labels[point] = cluster
            i = 0
            while i < len(NeighborPoints):
                point_n = NeighborPoints[i]
                if labels[point_n] == -1:
                    labels[point_n] = cluster
                elif labels[point_n] == 0:
                    labels[point_n] = cluster
                    NewNeighborPoints = getNeighbor(Datasets, point_n, eps)
                    if len(NewNeighborPoints) >= Minpoints:
                        NeighborPoints = NeighborPoints + NewNeighborPoints
                i += 1
    return labels

if __name__ == '__main__':
    #load the data
    epsilons = [5, 10]
    Minpoints = [5, 10]
    points = loadmat('DBSCAN.mat')['Points']

    for eps in epsilons:
        for minPts in Minpoints:
            print('eps =', eps, 'minPoints =', minPts, ':')
            labels = DBSCAN(points, eps, minPts)

            #print the result of each parameter setting
            clusterId = set(labels)
            clusters = []
            for i in clusterId:
                cluster = [index for index in range(len(labels)) if labels[index] == i]
                clusters.append(np.array(cluster))
                print('cluster', i, ':', len(cluster))

            #plot the dbsacn cluster result
            plt.title('DBSCAN of eps = ' + str(eps) + ' Minpoints = ' + str(minPts))
            color = ['pink', 'orange', 'purple', 'red', 'yellow', 'green']
            for i in range(len(clusters)-1):
                coordinates = points[clusters[i]]
                plt.scatter(coordinates[:, 0], coordinates[:, 1], c=color[i], label= i+1)

            #plot the outliers as blue
            noiseId = len(clusters)-1
            noise = points[clusters[noiseId]]
            plt.scatter(noise[:, 0], noise[:, 1], c='blue', label= -1)
            plt.legend()
            plt.show()



