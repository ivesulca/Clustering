import numpy as np
from sklearn.neighbors import kneighbors_graph

def kmeans(X:np.ndarray, k:int, centroids=None, tolerance=1e-2):

    dim = X.shape[1]
    n_obs = X.shape[0]

    if centroids == 'kmeans++':
        centroids = initial_centroid_kplus(X,k)
    else:
        centroids = np.random.rand(k,dim)

    new_centroids = np.random.rand(k,dim)
    centroid_assign=np.empty(n_obs)
    centroid_assign.fill(-1)

    diff_clusters = True

    while diff_clusters:

        # Finding nearest centroids
        m_distances=[]
        for j in range(0,k):
            dist_cluster = np.linalg.norm((X-centroids[j]),axis=1)
            m_distances.append(dist_cluster)

        centroid_assign = np.argmin(m_distances, axis=0)

        # Recomputing centroids
        for i_clust in range(0,k):
            new_clust = np.mean(X[np.where(centroid_assign==i_clust,True,False),:],axis=0)
            new_centroids[i_clust] = new_clust

        # Checking whether the new centroids are equal to the previous ones
        if np.all(np.isclose(centroids, new_centroids, atol=tolerance)):
            diff_clusters = False

        centroids = new_centroids

    # Mapping observations to each cluster
    #clusters =  np.array([np.where(centroid_assign ==i) for i in range(0,k)])
    clusters =  [(np.where(centroid_assign ==i))[0] for i in range(0,k)]
    clusters = np.array(clusters)
    return centroids, clusters

def initial_centroid_kplus(X,k):

    centroids = []
    selected_points = []
    # First random centroid
    first_point = np.random.choice(X.shape[0], 1)
    selected_points.append(first_point)
    centroids.append(X[first_point][0])

    for i in range(1,k):
        m_distances = []

        # Compute distances to each cluster
        n_k = len(centroids)
        for j in range(0,n_k):
            dist_cluster = np.linalg.norm((X-centroids[j]),axis=1)
            m_distances.append(dist_cluster)

        # Compute min distance to each cluster
        min_distances = np.amin(m_distances, axis = 0)

        # Choose the obs with the max distance
        i_next_cluster = np.argmax(min_distances)

        centroids.append(X[i_next_cluster])

    centroids = np.array(centroids)

    return centroids

def reshapeImg_2dmatrix(X, height, weight):
    X_r = np.reshape(X, (height*weight,3))
    return X_r.copy()

def reassign_colors(X,clusters,centroids):
    k = len(centroids)
    for i_cluster in range(0,k):
        #X[clusters[i_cluster][0]] = centroids[i_cluster]
        X[clusters[i_cluster]] = centroids[i_cluster]
    return X

def likely_predictions(y,clusters):

    y_pred = np.empty(len(y))
    commonpred_cluster = []
    k = clusters.shape[0]
    for c in range(0,k):
        targets=y[clusters[c]]
        #print(np.bincount(targets))
        most_common = np.argmax(np.bincount(targets))
        #print(most_common)
        commonpred_cluster.append(most_common)
        y_pred[clusters[c]] = most_common

    return y_pred

def eigenVectors(X,k_neighbors,k_clusters):
    #Use for spectral clustering

    # Using K-neighbors to transform raw matrix to a graph data
    adj_matrix = kneighbors_graph(X, n_neighbors=k_neighbors).toarray()
    b = np.diag(adj_matrix.sum(axis=1))
    laplacian_matrix = b - adj_matrix

    # Sorting eigen vectors in ascendent order. Ideally, we should pick the # of clusters based on the gap between
    # those eigen values, but for the report example we will choose always the number of clusters we already know.
    eig_values, eigen_vect = np.linalg.eig(laplacian_matrix)
    indexes_ascendent = np.argsort(eig_values)
    eigen_vect = eigen_vect[:,np.argsort(eig_values)]
    eigen_vect = eigen_vect.real

    return eigen_vect[:,0:(k_clusters)]
