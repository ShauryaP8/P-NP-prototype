import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx

def create_dataset(num_cities):
    """Generate synthetic dataset of cities with random coordinates."""
    coordinates = np.random.rand(num_cities, 2)  # 2D coordinates for the cities
    return coordinates

def apply_kmeans(coordinates, n_clusters):
    """Apply K-means clustering to partition the cities."""
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(coordinates)
    return clusters, kmeans

def create_geometric_graph(num_nodes, radius):
    """Create a random geometric graph based on node proximity."""
    G = nx.random_geometric_graph(num_nodes, radius)
    return G

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

# TSP solution using the nearest neighbor heuristic
def calculate_distance(city1, city2):
    return np.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2)

def nearest_neighbor_tour(coordinates):
    unvisited = set(range(len(coordinates)))
    tour = [unvisited.pop()]
    while unvisited:
        current_city = tour[-1]
        nearest_city = min(unvisited, key=lambda city: calculate_distance(coordinates[current_city], coordinates[city]))
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
    return tour

def main():
    num_cities = 100  # Number of cities
    n_clusters = 5   # Number of clusters for K-means
    radius = 0.125   # Radius for geometric graph connectivity

    # Create synthetic data
    coordinates = create_dataset(num_cities)

    # Apply k-means clustering
    clusters, kmeans_model = apply_kmeans(coordinates, n_clusters)

    # Create a geometric graph
    G = create_geometric_graph(num_cities, radius)

    # Visualize the clustering
    visualize_clusters(coordinates, clusters)

    # Optionally, visualize the graph - uncomment the following lines if needed
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_color=[G.nodes[data]['cluster'] for data in G.nodes], with_labels=True, cmap='viridis')

if __name__ == "__main__":
    main()
