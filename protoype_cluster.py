import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def create_dataset(num_cities):
    """Generate synthetic dataset of cities with random coordinates."""
    return np.random.rand(num_cities, 2)  # 2D coordinates for the cities

def apply_kmeans(coordinates, n_clusters):
    """Apply K-means clustering to partition the cities."""
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(coordinates)
    return clusters, kmeans

def visualize_clusters(coordinates, clusters):
    """Visualize the results of clustering."""
    plt.figure(figsize=(10, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
    plt.colorbar(label='Cluster ID')
    plt.title('Geographical Clustering of Cities')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

def calculate_distance(city1, city2):
    """Calculate Euclidean distance between two cities."""
    return np.linalg.norm(city2 - city1)

def nearest_neighbor_tour(coordinates):
    """Create a tour using the nearest neighbor heuristic."""
    unvisited = set(range(len(coordinates)))
    tour = [unvisited.pop()]
    while unvisited:
        current_city = tour[-1]
        nearest_city = min(unvisited, key=lambda city: calculate_distance(coordinates[current_city], coordinates[city]))
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
    return tour

def two_opt(tour, coordinates):
    """Refine tour using the 2-opt algorithm."""
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1: continue  # Skip adjacent edges
                new_distance = calculate_distance(coordinates[tour[i - 1]], coordinates[tour[j]]) + \
                               calculate_distance(coordinates[tour[i]], coordinates[tour[j + 1]])
                old_distance = calculate_distance(coordinates[tour[i - 1]], coordinates[tour[i]]) + \
                               calculate_distance(coordinates[tour[j]], coordinates[tour[j + 1]])
                if new_distance < old_distance:
                    tour[i:j] = tour[i:j][::-1]
                    improved = True
    return tour

def connect_clusters(clustered_tours, coordinates):
    """Connect cluster tours using nearest neighbor between cluster endpoints."""
    global_tour = list(clustered_tours[0])  # Start with the first cluster's tour
    for current_cluster in range(1, len(clustered_tours)):
        last_point = global_tour[-1]
        next_point = clustered_tours[current_cluster][0]
        global_tour += clustered_tours[current_cluster]

    # Apply 2-opt to the global tour
    return two_opt(global_tour, coordinates)

def main():
    num_cities = 100  # Number of cities
    n_clusters = 5    # Number of clusters for K-means

    # Create synthetic data
    coordinates = create_dataset(num_cities)

    # Apply k-means clustering
    clusters, kmeans_model = apply_kmeans(coordinates, n_clusters)

    # Visualize the clustering
    visualize_clusters(coordinates, clusters)

    # Generate tours for each cluster
    clustered_tours = {}
    for cluster_id in range(n_clusters):
        cluster_coordinates = coordinates[clusters == cluster_id]
        tour = nearest_neighbor_tour(cluster_coordinates)
        clustered_tours[cluster_id] = tour

    # Connect the tours into a global tour and apply 2-opt
    global_tour = connect_clusters(clustered_tours, coordinates)

    # Visualize the global optimized tour
    plt.figure(figsize=(10, 8))
    global_tour_coords = coordinates[global_tour]
    plt.plot(global_tour_coords[:, 0], global_tour_coords[:, 1], 'o-')  # Plot the tour
    plt.plot(global_tour_coords[0, 0], global_tour_coords[0, 1], 'ro')  # Highlight starting point
    plt.title('Global Optimized Tour')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
