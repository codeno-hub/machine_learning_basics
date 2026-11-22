# WEEK 12: SPECTRAL CLUSTERING, DBSCAN AND GMM

# PART 1: IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
print("All libraries imported successfully!")

# PART 2: UNDERSTANDING CLUSTERING
# ============================================================

# Create simple sample data
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
X
y

# Display original data
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Original Make Moons Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
print("Dataset shape:", X.shape)

# PART 3: SPECTRAL CLUSTERING
# ============================================================
# Step 1: Generate data
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_labels

# Step 4: Apply Spectral Clustering
spectral = SpectralClustering(
    n_clusters=2,
    affinity="rbf",
    gamma=0.5,
    random_state=42
)

spectral_labels = spectral.fit_predict(X_scaled)
spectral_labels

# Step 5: Visualize both methods
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)
plt.title("Original Data")

plt.subplot(1, 3, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels)
plt.title("K-Means")

plt.subplot(1, 3, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=spectral_labels)
plt.title("Spectral Clustering")

plt.tight_layout()
plt.show()

# Compare silhouette scores
print("K-Means Score:",
      round(silhouette_score(X_scaled, kmeans_labels), 3))
print("Spectral Score:",
      round(silhouette_score(X_scaled, spectral_labels), 3))

# PART 4: UNDERSTANDING SIMILARITY MATRIX
# ============================================================
# Calculate similarity between data points
from sklearn.metrics.pairwise import rbf_kernel

W = rbf_kernel(X_scaled, gamma=0.5)
print("Similarity Matrix Shape:", W.shape)

# Display similarity matrix
plt.figure(figsize=(6, 5))
plt.imshow(W, cmap="viridis")
plt.colorbar()
plt.title("Similarity Matrix")
plt.show()

# Create degree matrix
D = np.diag(np.sum(W, axis=1))
# Calculate Laplacian matrix
L = D - W
print("Laplacian Matrix Shape:", L.shape)

plt.figure(figsize=(6, 5))
plt.imshow(L, cmap="viridis")
plt.colorbar()
plt.title("Graph Laplacian Matrix")
plt.show()

# Find eigenvalues
eigenvalues = np.linalg.eigvalsh(L)
print("First 10 Eigenvalues:")
print(np.round(eigenvalues[:10], 3))

# PART 5: DBSCAN CLUSTERING
# Generate circular data
X, y = make_circles(
    n_samples=300,
    factor=0.5,
    noise=0.05,
    random_state=42
)
X, y

# Standardize the data
X_scaled = StandardScaler().fit_transform(X)
X_scaled

# Visualize original data-Example-Ex MRI/CT Image Segmentation
plt.figure(figsize=(5, 4))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1])
plt.title("Original Circles Dataset")
plt.show()

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
dbscan_labels

# Display clusters
plt.figure(figsize=(6, 4))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=dbscan_labels,
    cmap="viridis"
)

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Count clusters and noise
unique_labels = set(dbscan_labels)

print("Cluster Labels:", unique_labels)
print("Number of Clusters:",
      len(unique_labels - {-1}))
print("Noise Points:",
      np.sum(dbscan_labels == -1))

# ============================================================
# PART 6: DBSCAN PARAMETER TUNING
# ============================================================

# Try different epsilon values
eps_values = [0.1, 0.2, 0.3, 0.5]
for eps in eps_values:

    model = DBSCAN(eps=eps, min_samples=5)

    labels = model.fit_predict(X_scaled)

    clusters = len(set(labels) - {-1})
    noise = np.sum(labels == -1)

    print("\nEpsilon:", eps)
    print("Clusters:", clusters)
    print("Noise Points:", noise)

    plt.figure(figsize=(5, 3))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
    plt.title("DBSCAN with eps = " + str(eps))
    plt.show()

# ============================================================
# PART 7: K-DISTANCE GRAPH
# ============================================================
from sklearn.neighbors import NearestNeighbors

# Find distance to the 5th nearest neighbor
neighbors = NearestNeighbors(n_neighbors=5)

neighbors.fit(X_scaled)

distances, indices = neighbors.kneighbors(X_scaled)

# Sort distances
k_distances = np.sort(distances[:, 4])

# Plot k-distance graph
plt.figure(figsize=(7, 4))

plt.plot(k_distances)

plt.xlabel("Data Points")
plt.ylabel("5th Nearest Neighbor Distance")

plt.title("DBSCAN K-Distance Graph")
plt.show()

print("Use the elbow of the graph to help choose epsilon.")



"""# GMM Model on IRIS Data"""

# PART 8: GAUSSIAN MIXTURE MODEL (GMM)
# ============================================================

# Load Iris dataset
iris = load_iris()

# Take first two features
X = iris.data[:, :2]

# Standardize data
X_scaled = StandardScaler().fit_transform(X)

# Build GMM model
gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    max_iter=300,
    reg_covar=1e-6,
    random_state=42
)

# Train model
gmm.fit(X_scaled)

# Predict cluster labels
gmm_labels = gmm.predict(X_scaled)

# Display clusters
plt.figure(figsize=(6, 4))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=gmm_labels,
    cmap="viridis"
)

plt.title("GMM Clustering on Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

print("GMM Cluster Labels:")
print(gmm_labels[:20])

# ============================================================
# PART 9: GMM SOFT CLUSTERING
# ============================================================

# Calculate probability of belonging to each cluster
probabilities = gmm.predict_proba(X_scaled)

print("First 5 Probability Predictions:")
print(np.round(probabilities[:5], 3))

# Find the most uncertain observations
# Lower maximum probability means greater uncertainty
uncertainty = np.max(probabilities, axis=1)

most_uncertain = np.argsort(uncertainty)[:5]

print("\nMost Uncertain Observations:")
print(most_uncertain)

print("\nTheir Cluster Probabilities:")
print(np.round(probabilities[most_uncertain], 3))

# ============================================================
# PART 10: GMM EM CONVERGENCE
# ============================================================
# Train GMM again and store likelihood at each iteration
gmm = GaussianMixture(
    n_components=3,
    max_iter=300,
    tol=1e-4,
    reg_covar=1e-6,
    random_state=42
)
gmm.fit(X_scaled)
print("Number of EM Iterations:", gmm.n_iter_)
print("Converged:", gmm.converged_)
print("Final Log Likelihood:", gmm.lower_bound_)

# Plot likelihood history
history = []
gmm2 = GaussianMixture(n_components=3, covariance_type='full',
                       random_state=42, max_iter=1, warm_start=True)

for i in range(100):
    gmm2.fit(X_scaled)
    history.append(gmm2.lower_bound_)  # singular, this exists
    if len(history) > 1 and abs(history[-1] - history[-2]) < 1e-4:
        break

print(f"Converged in {len(history)} iterations")
print(f"Final Log-Likelihood: {history[-1]:.3f}")

plt.figure(figsize=(8,5))
plt.plot(range(1, len(history)+1), history, marker='o', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood')
plt.title('GMM EM Convergence')
plt.grid(True, alpha=0.3)
plt.show()

# PART 11: COMPARING ALL THREE ALGORITHMS
# ============================================================

# Use one common dataset for comparison
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

X_scaled = StandardScaler().fit_transform(X)

# Spectral Clustering
spectral = SpectralClustering(
    n_clusters=2,
    affinity="rbf",
    gamma=0.5,
    random_state=42
)

spectral_labels = spectral.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.25, min_samples=5)

dbscan_labels = dbscan.fit_predict(X_scaled)

# GMM
gmm = GaussianMixture(
    n_components=2,
    reg_covar=1e-6,
    random_state=42
)

gmm_labels = gmm.fit_predict(X_scaled)

# Plot all three
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=spectral_labels)
plt.title("Spectral Clustering")

plt.subplot(1, 3, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=dbscan_labels)
plt.title("DBSCAN")

plt.subplot(1, 3, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=gmm_labels)
plt.title("GMM")

plt.tight_layout()
plt.show()
