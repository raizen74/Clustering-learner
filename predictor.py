# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 13:08:31 2022

@author: David
"""

import csv

import numpy as np
import pandas as pd

from learner import get_closest_centroid

# Llegim les dades i les guardem en arrays
dades = []

with open("param.out", "r") as fd:
    reader = csv.reader(fd)
    for row in reader:
        dades.append(row)

n_clusters = int(dades[0][0])
centroids = []
for i in range(1, n_clusters + 1):
    centroids.append(list(map(float, dades[i][0].split(" "))))
centroids = np.array(centroids)
means = np.array(list(map(float, dades[n_clusters + 1][0].split(" "))))
variances = np.array(list(map(float, dades[n_clusters + 2][0].split(" "))))
components = []
for i in range(n_clusters + 3, len(dades)):
    components.append(list(map(float, dades[i][0].split(" "))))
components = np.array(components)

# Test
test = pd.read_csv("testing.csv", header=None)
test = np.array(test.values.tolist())

# Estandaritzem i transformem dades al mateix espai PCA que el del training.csv
test_standar = np.array([(x - means) / (variances**0.5) for x in test])
test_transformed = np.matmul(test_standar, components.T)

# Assignem a cada observació el centroide més proper
label = list(map(lambda x: get_closest_centroid(x, centroids), test_transformed))

# Mapem per a cada centroide el número de cluster que li pertany
for i in range(len(label)):
    for index, x in enumerate(centroids):
        if np.array_equal(x, label[i]):
            label[i] = index

# Visualització 2 PCA

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.style.use('seaborn')
# ax = sns.scatterplot(x=test_transformed[:, 0], y=test_transformed[:, 1], hue=label)
# ax = sns.scatterplot(x=centroids[:,0],y=centroids[:,1],c='red',s=100)
# ax.set_title('Standarized 2 PCA', fontsize=16);
# plt.show()

# Exportem els resultats del clustering
with open("clustering.out", "w") as my_file:
    np.savetxt(my_file, label, fmt="%i")
print("Array exported to file")
