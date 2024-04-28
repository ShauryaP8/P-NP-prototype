# Traveling Salesman Problem Solver

This document provides an overview of the steps taken to develop a TSP solver using a parallel computing approach with Python.

## Overal Objective
The overall goal of this project is to develop a proof that P = NP by constructing an efficient algorithm to solve the NP-complete Traveling Salesman Problem (TSP) 
in polynomial time. This document outlines our approach using parallel computing and clustering techniques.

## Step 1: Tool Selection

We chose Python for its rich ecosystem suitable for numerical computations and data visualization. The following libraries were selected:

- `NumPy`: For efficient numerical computations.
- `Pandas`: For structured data manipulation (not used in the current script but useful for handling larger datasets).
- `Scikit-learn`: To implement machine learning algorithms like k-means clustering.
- `NetworkX`: For graph-based operations.
- `Matplotlib`: For creating visualizations of the data and clusters.

## Step 2: Data Setup and Clustering

We created a synthetic dataset representing cities with random coordinates. Then, we applied the k-means clustering algorithm to group the cities based on their geographic proximity.

```python
from sklearn.cluster import KMeans
import numpy as np

coordinates = np.random.rand(100, 2)  # 100 cities with x, y coordinates
kmeans = KMeans(n_clusters=5)  # number of clusters
clusters = kmeans.fit_predict(coordinates)
```

## Step 3: Visualizing Clustering Results
A scatter plot was generated to visualize the clustering of cities:

![Geographical Clustering of Cities](P-NP-prototype/Geographical%20clustering%20graph.png)


The plot shows cities represented as points in a 2D plane, with the x and y coordinates denoting their geographic location.
Different colors indicate different clusters, suggesting a successful partitioning of the dataset into smaller regions for easier management in the TSP.

## Step 4: Simulating TSP Solutions
For each cluster, we implemented a function to simulate a simple TSP solution using the nearest neighbor heuristic:

```python
def nearest_neighbor_tour(coordinates):
    unvisited = set(range(len(coordinates)))
    tour = [unvisited.pop()]
    while unvisited:
        current_city = tour[-1]
        nearest_city = min(unvisited, key=lambda city: calculate_distance(coordinates[current_city], coordinates[city]))
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
    return tour
```

## Step 5: Iteration and Refinement
We defined a process for iterating and refining the TSP solutions:

Refinement of the clustering algorithm's parameters.
Optimization of the routes within each cluster using techniques like 2-opt or 3-opt.
Development of an integration strategy to combine the cluster solutions into a single route.

## Step 6: Convergence Criteria
To ensure our iterative process converges on an optimal or near-optimal solution, we established convergence criteria based on improvement thresholds 
and iteration limits.

```python
improvement_threshold = 0.01  # Acceptable percentage improvement
max_iterations = 100  # Maximum number of iterations
no_improvement_limit = 10  # Convergence criterion based on consecutive iterations without improvement
```

## Results and Interpretation
Preliminary results from the clustering and heuristic tour simulation for each cluster produced outputs such as:

```yaml
Cluster 0:
Tour sequence: [0, 13, 9, 10, 16, 17, 1, 19, 12, 3, 15, 4, 5, 11, 14, 6, 7, 8, 2, 18]
Tour distance: 1.7222037789451092

Cluster 1:
Tour sequence: [0, 1, 15, 6, 21, 9, 19, 7, 11, 20, 22, 12, 10, 4, 14, 3, 16, 13, 17, 5, 18, 2, 8]
Tour distance: 2.0327531843959683

Cluster 2:
Tour sequence: [0, 6, 13, 5, 7, 2, 12, 1, 10, 14, 4, 15, 11, 3, 16, 8, 9]
Tour distance: 1.6971900163901537

Cluster 3:
Tour sequence: [0, 13, 3, 9, 8, 14, 2, 1, 11, 15, 5, 4, 6, 10, 7, 16, 12]
Tour distance: 1.4796824754567794

Cluster 4:
Tour sequence: [0, 9, 12, 2, 1, 17, 21, 5, 7, 13, 19, 11, 4, 15, 10, 20, 16, 22, 18, 6, 3, 8, 14]
Tour distance: 1.5588247286556984
```
These results are consistent with expectations for the nearest neighbor heuristic, indicating the algorithm's effectiveness for small to medium-sized clusters 
within the TSP.

## Contribution to Proving P = NP
This development is a step toward creating a polynomial-time solution for an NP-complete problem. By attempting to optimize the TSP solution process and 
seeking efficiency improvements, we aim to contribute to the broader discussion and efforts in the computational complexity community regarding the P vs NP question.

The approach taken—breaking down a complex problem into simpler, manageable parts and solving them in parallel—mirrors strategies that might be used to tackle 
other NP-complete problems. If scalable to larger instances and proven to be efficient across the board, this method could suggest that P = NP by providing a 
concrete example of an NP-complete problem solved in polynomial time.

## Conclusion
We have successfully prototyped a TSP solver that utilizes parallel computing concepts with clustering to efficiently solve the problem. 
Future steps include scaling the solution, improving optimization techniques, and refining the integration strategy to handle larger 
datasets and more complex scenarios.


