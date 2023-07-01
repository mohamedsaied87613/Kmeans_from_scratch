import numpy as np
import random
from tqdm import tqdm

# pick random centroids with number of clusters:
def get_centroids(data, n=3):
    random_values = random.choices(range(0, data.shape[0]), k=n)
    return np.array(data.loc[random_values])


# define cost function for kmeans (MSE)
def cost(data, clusters, centroids):
    cost = [np.sum(np.linalg.norm(data.iloc[np.where(clusters == i)[0]] - centroids[i], axis=1) ** 2) for i in
            range(centroids.shape[0])]
    return np.sum(cost)


def train(data, centroids, epochs):
    loss = []
    precentage_of_clusters = []
    for i in tqdm(range(epochs)):
        # get distance between each point and centroids
        dist = np.array([np.sum(np.sqrt((data - j) ** 2), axis=1) for j in centroids])

        # select which one is closer
        clusters = np.argmin(dist, axis=0)

        precentage_of_clusters.append(
            [(data.iloc[np.where(clusters == i)]).shape[0] / data.shape[0] * 100 for i in range(centroids.shape[0])])

        # update centroids
        centroids = np.array([data.iloc[np.where(clusters == i)].sum() / np.where(clusters == i)[0].shape[0] for i in
                              range(centroids.shape[0])])

        loss.append(cost(data, clusters, centroids))

    return clusters, centroids, loss, precentage_of_clusters
