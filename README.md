**Facebook Sales in Thailand Region using K-Means Clustering**

In this project, I applied K-Means clustering for customer segmentation. The focus was on determining the optimal number of clusters using the Elbow Method, visualizing centroids, and evaluating model performance with the Silhouette Plot.

**1. Elbow Method**

The Elbow Method was used to determine the optimal number of clusters by plotting the sum of squared distances (inertia) against a range of cluster values.
```import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import numpy as np

# Membuat dataset contoh
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fungsi untuk menghitung dan menampilkan hasil Elbow Method dengan WCSS (Elbow Score)
def elbow_method(X):
    wcss = []  # List untuk menyimpan WCSS untuk berbagai k
    for i in range(1, 11):  # Menguji k dari 1 hingga 10
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)  # Menyimpan WCSS (inertia)

    # Menampilkan Elbow Score (WCSS) untuk setiap k
    print("Elbow Scores (WCSS) untuk setiap k:")
    for i, score in enumerate(wcss, start=1):
        print(f"k={i}: WCSS = {score:.2f}")

    # Plot Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', color='blue')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.show()

    # Mengembalikan WCSS untuk analisis lebih lanjut
    return wcss

# Fungsi untuk menghitung dan menampilkan Silhouette Index untuk berbagai nilai k
def silhouette_index(X):
    sil_scores = []  # List untuk menyimpan Silhouette Score untuk setiap k
    for i in range(2, 11):  # Silhouette hanya dapat dihitung untuk k >= 2
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(X)

        # Menghitung Silhouette Score untuk seluruh klaster
        sil_score = silhouette_score(X, cluster_labels)
        sil_scores.append(sil_score)

    # Plot Silhouette Score untuk memilih k optimal
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), sil_scores, marker='o', color='green')
    plt.title('Silhouette Score for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.show()

    # Mengembalikan Silhouette Scores untuk analisis lebih lanjut
    return sil_scores

# Menjalankan fungsi Elbow Method
wcss = elbow_method(X)

# Menjalankan fungsi Silhouette Index
sil_scores = silhouette_index(X)

# Menentukan nilai optimal k berdasarkan Elbow dan Silhouette Score
optimal_k_elbow = np.argmin(np.diff(np.diff(wcss))) + 2  # Menemukan titik Elbow
optimal_k_silhouette = np.argmax(sil_scores) + 2  # Menemukan nilai k dengan Silhouette Score tertinggi

print(f"Nilai k optimal berdasarkan Elbow Method: {optimal_k_elbow}")
print(f"Nilai k optimal berdasarkan Silhouette Index: {optimal_k_silhouette}")
 ```

![tugas 2_page-0001](https://github.com/user-attachments/assets/d50e29d5-e6b3-4bd7-976c-4380fd0e874e)

**2. Visualization Using Euclidean Distance**

Clusters were visualized using Euclidean distance for cluster assignments.
```import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

cleaned_data = dataset[['num_shares', 'num_comments','Cluster']]
X = cleaned_data[['num_shares', 'num_comments','Cluster']]

# Pilih kolom numerik untuk clustering
numerical_cols = cleaned_data.select_dtypes(include=['float64', 'int64']).columns
numerical_data = cleaned_data[numerical_cols]

# Inisialisasi parameter
n_clusters = 3
max_iter = 100
tolerance = 1e-4  # Toleransi perubahan centroid
previous_centroids = None

# Loop untuk memastikan centroid stabil
for iteration in range(max_iter):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1, max_iter=1, init='random')
    kmeans.fit(numerical_data)
    
    current_centroids = kmeans.cluster_centers_
    if previous_centroids is not None:
        centroid_shift = np.linalg.norm(current_centroids - previous_centroids)
        print(f"Iterasi {iteration + 1}, Perubahan Centroid: {centroid_shift}")
        if centroid_shift < tolerance:
            print("Centroid stabil, berhenti iterasi.")
            break
    previous_centroids = current_centroids

# Tambahkan kolom hasil clustering
cleaned_data['Cluster'] = kmeans.predict(numerical_data)

# Scatter plot untuk visualisasi clustering
plt.figure(figsize=(12, 8))
if len(numerical_cols) >= 2:
    scatter_x = numerical_data[numerical_cols[0]]
    scatter_y = numerical_data[numerical_cols[1]]

    sns.scatterplot(
        x=scatter_x, 
        y=scatter_y, 
        hue=cleaned_data['Cluster'], 
        palette='Set1', 
        s=70, 
        style=cleaned_data['Cluster'], 
        markers=["o", "s", "D"]
    )

    plt.scatter(current_centroids[:, 0], current_centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
    plt.title(f'Visualisasi Clustering Stabil menggunakan {numerical_cols[0]} dan {numerical_cols[1]}')
    plt.xlabel(numerical_cols[0])
    plt.ylabel(numerical_cols[1])
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()
else:
    print("Tidak cukup kolom numerik untuk scatter plot.")

 ```

![tugas 2_page-0003](https://github.com/user-attachments/assets/17482299-4c1f-4ec7-a4cc-c4ad26fe10c1)

**3. Evaluation Using Silhouette Plot**

The Silhouette Plot was used to evaluate clustering performance.

```import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np

cleaned_data = dataset[['num_shares', 'num_comments','Cluster']]
X = cleaned_data[['num_shares', 'num_comments','Cluster']]

# Pilih kolom numerik untuk clustering
numerical_cols = cleaned_data.select_dtypes(include=['float64', 'int64']).columns
numerical_data = cleaned_data[numerical_cols]

# Clustering dengan K-Means
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(numerical_data)

# Hitung Silhouette Score
silhouette_avg = silhouette_score(numerical_data, cluster_labels)
print(f"Silhouette Score untuk {n_clusters} cluster: {silhouette_avg:.4f}")

# Visualisasi Silhouette Score
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
silhouette_vals = silhouette_samples(numerical_data, cluster_labels)
y_lower, y_upper = 0, 0
for i in range(n_clusters):
    cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    color = plt.cm.Spectral(float(i) / n_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7, color=color)
    ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i))
    y_lower = y_upper
ax.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.title(f"Silhouette Plot untuk {n_clusters} Cluster")
plt.xlabel("Nilai Silhouette")
plt.ylabel("Cluster")
plt.grid(True)
plt.show()

# Plot jumlah cluster dan nilai Silhouette Score untuk evaluasi
plt.figure(figsize=(8, 5))
cluster_range = range(2, 6)
silhouette_scores = []
for n in cluster_range:
    kmeans = KMeans(n_clusters=n, random_state=42)
    labels = kmeans.fit_predict(numerical_data)
    score = silhouette_score(numerical_data, labels)
    silhouette_scores.append(score)

plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title("Silhouette Score untuk Berbagai Jumlah Cluster")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()
 ```

![tugas 2_page-0004](https://github.com/user-attachments/assets/0a915fac-d489-4935-8f17-fd85ca1ee8f7)

**Results**

The Elbow Method indicated an optimal cluster number at k=3.
Euclidean distance-based visualization illustrated refined cluster boundaries.
Silhouette Plot confirmed clustering quality with an average score of 0.8

**Conclusion**

This project demonstrates effective customer segmentation using K-Means clustering. Future work may include scaling data and testing additional algorithms for comparison.

