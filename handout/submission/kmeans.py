import numpy as np

'''def normalize(data):
    maxes = [data[:, col].max() for col in range(data.shape[1])]
    for col, max in enumerate(maxes):
        data[:,col] /= max
    return maxes

def unnormalize(data, maxes):
    for col, max in enumerate(maxes):
        data[:,col] *= max
    return data
'''
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
    #maxes = normalize(data)
    # Create a random number generator
    # Use this to avoid fluctuation in k-means performance due to initialisation
    rng = np.random.default_rng(6174)
    
    # Initialise clusters
    centroids = data[rng.choice(N, k, replace=False)]
    
    # Iterate the k-means update steps
    change = True
    count = 0
    while change and count < n_iter:
        count += 1
        # Initialize the cluster accumulators
        centers = np.zeros_like(centroids)
        # We use counts so we can divide by it to get the mean
        counts = np.zeros((centers.shape[0],))

        distances = compute_distance(data, centroids)
        #print(f"dist shape: {distances.shape}")
        # Pick the closest centroid, i.e. every sample gets an index
        index = np.argmin(distances, axis=-1)
        
        # Iterate over all samples
        for i in range(N):
            # Add the data point to the corresponding centroid accumulator.
            cluster = index[i]
            centers[cluster] += data[i]
            counts[cluster] += 1
        
        # Make the average of the points.
        #print(centers.shape)
        #print(counts.shape)
        for c in range(centers.shape[0]):
            centers[c] /= counts[c]

        # Check if the new centers are far enough away from the old centers.
        change = np.any(np.linalg.norm(centroids - centers) > tol)
        if change:
            centroids = centers

    # Return cluster centers
    # unnormalize(centroids, maxes)
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