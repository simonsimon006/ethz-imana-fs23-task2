import numpy as np


def kmeans_fit(data, k, n_iter=500, tol=1.e-4):
    """
    Fit kmeans
    
    Args:
        data ... Array of shape n_samples x n_features
        k    ... Number of clusters
        
    Returns:
        centers   ... Cluster centers. Array of shape k x n_features
    """
    N, P = data.shape
    
    # Create a random number generator
    # Use this to avoid fluctuation in k-means performance due to initialisation
    rng = np.random.default_rng(6174)
    
    # Initialise clusters
    centroids = data[rng.choice(N, k, replace=False)]
    
    # Iterate the k-means update steps
    change = True
    while change:

        # Initialize the cluster accumulators
        centers = np.zeros_like(centroids)
        # We use counts so we can divide by it to get the mean
        counts = np.zeros(centroids.shape[0])

        # Iterate over all samples
        for i in range(N):
            # Calculate the distances to all centroids. The order is important.
            distances = []
            for center in centroids:
                distances.append(np.linalg.norm(data[i] - center))
            # Pick the closest centroid.
            index = np.argmin(distances)
            # Add the data point to the corresponding centroid accumulator.
            centers[index] += data[i]
            counts[i] += 1
        
        # Make the average of the points.
        centers /= counts

        # Check if the new centers are far enough away from the old centers.
        change = np.linalg.norm(centroids - centers) > 1e-5
        if change:
            centroids = centers

    # Return cluster centers
    return centroids


def compute_distance(data, clusters):
    """
    Compute all distances of every sample in data, to every center in clusters.
    
    Args:
        data     ... n_samples x n_features
        clusters ... n_clusters x n_features
        
    Returns:
        distances ... n_samples x n_clusters
    """

    # I will not do the numpy concat shitfuckery.
    result = np.zeros((data.shape[0], clusters.shape[0]))

    for i in range(data.shape[0]):
        for c in range(clusters.shape[0]):
            result[i, c] = np.linalg.norm(data[i] - clusters[c])
    # TO IMPLEMENT
    return result


def kmeans_predict_idx(data, clusters):
    """
    Predict index of closest cluster for every sample
    
    Args:
        data     ... n_samples x n_features
        clusters ... n_clusters x n_features
    """
    distances = compute_distance(data, clusters)
    return np.argmin(distances, axis=-1)