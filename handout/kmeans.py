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
    #
    # TO IMPLEMENT
    #
            
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
    
    # TO IMPLEMENT
    return -1


def kmeans_predict_idx(data, clusters):
    """
    Predict index of closest cluster for every sample
    
    Args:
        data     ... n_samples x n_features
        clusters ... n_clusters x n_features
    """
    # TO IMPLEMENT