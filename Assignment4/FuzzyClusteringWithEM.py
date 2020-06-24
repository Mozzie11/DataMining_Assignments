from scipy import io
import numpy as np
import math
import matplotlib.pyplot as plt

def distance(x,y):
    #euclidean distance
    distance = math.pow(x[0]-y[0], 2) + math.pow(x[1]-y[1], 2)
    return distance

def SSE(center, object, w):
    #sum of squared error
    sum = 0
    for i in range(len(object)):
        for j in range(len(center)):
            sum += distance(object[i], center[j]) * w[i, j]
    return sum

def EM(object, iteration):
    #initiation
    center = []
    cluster1 = []
    cluster2 = []
    partition_matrix = np.zeros((2, len(object)))
    #make the first two points as c1 c2
    for i in range(2):
        center.append(object[i])
    center = np.array(center)

    for iter in range(iteration):
        #E-step
        cluster1 = []
        cluster2 = []
        for i in range(len(object)):
            distance_c1 = distance(object[i], center[0])
            distance_c2 = distance(object[i], center[1])
            partition_matrix[0, i] = distance_c2 / (distance_c2 + distance_c1)
            partition_matrix[1, i] = 1 - partition_matrix[0, i]
            if distance_c1 < distance_c2:
                cluster1.append(object[i])
            else:
                cluster2.append(object[i])

        #M-step
        center[0] = np.square(partition_matrix).dot(object)[0] / np.sum(np.square(partition_matrix), axis=1)[0]
        center[1] = np.square(partition_matrix).dot(object)[1] / np.sum(np.square(partition_matrix), axis=1)[1]

        sse = SSE(center, object, partition_matrix.transpose())
        print('Iteration', iter, 'center of cluster:', center[0], center[1], 'SSE:', sse)
    return np.array(cluster1), np.array(cluster2)

if __name__ == '__main__':
    #load the data
    mat = io.loadmat('EM_Points.mat')
    points = mat['Points']
    coordinates = points[:, 0:2]
    labels = points[:, 2]

    #set the iteration to do clustering
    iteration = 12
    cluster1, cluster2 = EM(coordinates, iteration)

    #plot the original cluster
    plt.title('original cluster')
    plt.scatter(points[:, 0], points[:, 1], s=20, c=points[:, 2])
    plt.show()

    #plot the fuzzy cluster
    plt.title('fuzzy cluster')
    plt.scatter(cluster1[:, 0], cluster1[:, 1], s=20, c='green')
    plt.scatter(cluster2[:, 0], cluster2[:, 1], s=20, c='red')
    plt.show()




