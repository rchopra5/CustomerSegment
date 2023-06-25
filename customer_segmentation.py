import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load customer data
customer_data = pd.read_csv('customer_data.csv')

# Perform data preprocessing
# Select relevant features for clustering
selected_features = ['Age', 'Income']
data = customer_data[selected_features]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Determine the optimal number of clusters using the elbow method
inertia = []
max_clusters = 10
for k in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, max_clusters + 1), inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()

# Perform K-means clustering with the chosen number of clusters
num_clusters = 4  # Set the desired number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_data)

# Add cluster labels to the original data
customer_data['Cluster'] = kmeans.labels_

# Analyze the results
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_labels = pd.Series(kmeans.labels_, name='Cluster')
customer_data_with_clusters = pd.concat([customer_data, cluster_labels], axis=1)

# Visualize the clusters
plt.scatter(customer_data['Age'], customer_data['Income'], c=customer_data['Cluster'])
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Customer Segmentation')
plt.show()

# Print the cluster centers
print("Cluster Centers:")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i+1}: {center}")

# Print the assigned clusters for each customer
print("\nCustomer Clusters:")
print(customer_data_with_clusters[['CustomerID', 'Cluster']])
