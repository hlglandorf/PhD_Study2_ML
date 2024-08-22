import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Import imputed data
df = pd.read_csv("s2w1imputed.csv")

# DBSCAN prep
# Find best epsilon via nearest neighbor algorithm
nearest_neighbors = NearestNeighbors(n_neighbors=15)
neighbors = nearest_neighbors.fit(df)

distances, indices = neighbors.kneighbors(df)
distances = np.sort(distances[:,14], axis=0)

i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")
#plt.show()

print(distances[knee.knee])
best_eps = distances[knee.knee] # best epsilon from above algorithm

# DBSCAN
# Now, using the epsilon for the algorithm
dbscan_cluster1 = DBSCAN(eps=best_eps, min_samples=14)
dbscan_cluster1.fit(df)

# Number of clusters
labels=dbscan_cluster1.labels_
N_clus=len(set(labels))-(1 if -1 in labels else 0)
print('Estimated no. of clusters: %d' % N_clus)

# Identify noise
n_noise = list(dbscan_cluster1.labels_).count(-1)
print('Estimated no. of noise points: %d' % n_noise)

# Add labels to the dataframe
df['DB_Cluster'] = labels
#print(df.T) # -1 is the identifier for noise and can therefore be removed

# Remove noise from dataframe
dfc = df.loc[df['DB_Cluster'] == 0]
print(dfc.T)

# Write out datafile to use in subsequent analyses
dfc.to_csv('s2w1impdb.csv', index=False)
