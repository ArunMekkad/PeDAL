import numpy as np
import os

def read_point_cloud(input_file):
    """
    Reads a .bin file containing 3D point cloud data (x, y, z, R).
    
    Parameters:
    - input_file: str, path to the .bin file.

    Returns:
    - numpy.ndarray: Array of shape (N, 4) with columns (x, y, z, R).
    """
    try:
        data = np.fromfile(input_file, dtype=np.float32)
        if len(data) % 4 != 0:
            raise ValueError("Invalid .bin file: number of elements is not a multiple of 4.")
        return data.reshape((-1, 4))
    except Exception as e:
        print(f"Error reading point cloud: {e}")
        return None

def rectangleFilter(points, min_x=None, max_x=None, min_z=None, max_z=None):
    mask = np.ones(points.shape[0], dtype=bool)
    if min_x is not None:
        mask &= points[:, 0] >= min_x
    if max_x is not None:
        mask &= points[:, 0] <= max_x
    
    if min_z is not None:
        mask &= points[:, 2] >= min_z
    if max_z is not None:
        mask &= points[:, 2] <= max_z
    
    return points[mask]


#points__ = rectangleFilter(points,min_x=-7, max_x=-1-0.13, min_z=0, max_z=3)
#points__ = rectangleFilter(points,min_x= 1 - 0.13, max_x=7, min_z=0, max_z=3)
    
import numpy as np
from sklearn.cluster import DBSCAN

def filter_clusters(points, min_points, max_points, eps=0.5, min_samples=5):
    """
    Filters 3D point cloud clusters based on their size and prints core metrics.
    
    Args:
        points (list of list or ndarray): List of 3D points [[x, y, z], ...].
        min_points (int): Minimum number of points required in a cluster.
        max_points (int): Maximum number of points allowed in a cluster.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        ndarray: Filtered points belonging to valid clusters.
        dict: Metrics including cluster sizes and number of clusters.
    """
    # Convert points to a NumPy array
    points = np.array(points)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    
    # Get cluster labels
    labels = clustering.labels_
    
    # Identify unique clusters and filter by size
    filtered_points = []
    cluster_sizes = {}
    unique_labels = set(labels) - {-1}  # Exclude noise points labeled as -1
    for label in unique_labels:
        cluster_points = points[labels == label]
        cluster_size = len(cluster_points)
        cluster_sizes[label] = cluster_size
        if min_points <= cluster_size <= max_points:
            filtered_points.extend(cluster_points)
    
    # Print core metrics
    num_clusters = len(cluster_sizes)
    print(f"Number of clusters: {num_clusters}")
    print(f"Cluster sizes: {cluster_sizes}")
    print(f"Filtered clusters: {len(filtered_points)} points retained")

    
    return np.array(filtered_points)
def filter_point_cloud(points):

    points__ = rectangleFilter(points,min_x=-7, max_x=7, min_z=0, max_z=3)
    points__ = filter_clusters(points__, min_points=2, max_points=200, eps=5, min_samples=20)

    return points__


def save_point_cloud(output_file, points):
    """
    Saves a 3D point cloud to a .bin file.

    Parameters:
    - output_file: str, path to save the .bin file.
    - points: numpy.ndarray, point cloud of shape (N, 4) with columns (x, y, z, R).
    """
    try:
        points.astype(np.float32).tofile(output_file)
        print(f"Point cloud saved to: {output_file}")
    except Exception as e:
        print(f"Error saving point cloud: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read the point cloud
    point_cloud = read_point_cloud(input_file)
    if point_cloud is None:
        sys.exit(1)

    # Apply filtration (example: filter points within a bounding box and intensity range)
    filtered_points = filter_point_cloud(point_cloud)

    print(f"Original point count: {point_cloud.shape[0]}, Filtered point count: {filtered_points.shape[0]}")

    # Save the filtered point cloud
    save_point_cloud(output_file, filtered_points)
