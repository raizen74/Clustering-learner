# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:01:48 2022

@author: David
"""

import sys

import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial import distance
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


def get_closest_centroid(obs, centroids):
    """
    Function for retrieving the closest centroid to the given observation
    in terms of the Euclidean distance.

    Parameters
    ----------
    obs : array
        An array containing the observation to be matched to the nearest centroid
    centroids : array
        An array containing the centroids

    Returns
    -------
    min_centroid : array
        The centroid closes to the obs
    """
    min_distance = sys.float_info.max
    min_centroid = 0

    for c in centroids:
        dist = distance.euclidean(obs, c)
        if dist < min_distance:
            min_distance = dist
            min_centroid = c

    return min_centroid


def get_prediction_strength(k, train_centroids, x_test, test_labels):
    """
    Function for calculating the prediction strength of clustering

    Parameters
    ----------
    k : int
        The number of clusters
    train_centroids : array
        Centroids from the clustering on the training set
    x_test : array
        Test set observations
    test_labels : array
        Labels predicted for the test set

    Returns
    -------
    prediction_strength : float
        Calculated prediction strength
    """
    n_test = len(x_test)
    """
    Omplim la co-membership matrix, això és: Per a totes les combinacions de parelles
    d'observacions del test set evaluem si cauen dins un mateix centroide del training set.
    En cas que hi caiguin: D[c1,c2] = 1.0
    """
    # populate the co-membership matrix
    D = np.zeros(shape=(n_test, n_test))
    for x1, c1 in zip(x_test, list(range(n_test))):
        for x2, c2 in zip(x_test, list(range(n_test))):
            if tuple(x1) != tuple(x2):
                if tuple(get_closest_centroid(x1, train_centroids)) == tuple(
                    get_closest_centroid(x2, train_centroids)
                ):
                    D[c1, c2] = 1.0

    # calculate the prediction strengths for each cluster
    ss = []
    for j in range(k):
        s = 0
        examples_j = x_test[test_labels == j, :].tolist()
        n_examples_j = len(examples_j)
        if n_examples_j > 1:
            for x1, l1, c1 in zip(x_test, test_labels, list(range(n_test))):
                for x2, l2, c2 in zip(x_test, test_labels, list(range(n_test))):
                    if tuple(x1) != tuple(x2) and l1 == l2 and l1 == j:
                        s += D[c1, c2]
            ss.append(s / (n_examples_j * (n_examples_j - 1)))

    prediction_strength = min(ss)

    return prediction_strength


df = pd.read_csv("training.csv", header=None)
# Calculem el valor crític de la mahalanobis distance amb un nivell de significancia del 0.01, assumint que la t-statistic segueix una distribució chi-square
outliers = chi2.ppf((1 - 0.01), df=max(df.columns))
df["mahala"] = mahalanobis(x=df, data=df)
df.drop(df.loc[df.mahala > outliers].index, axis=0, inplace=True)  # Eliminem outliers
df.drop("mahala", axis=1, inplace=True)
df = np.array(df.values.tolist())  # Convertim el dataframe a array

scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(df)

# Calculem el numero mínim de PC que expliquin el 90% de variança, mínim 2 PC
ncomp = (
    min(
        [
            index
            for index, x in enumerate(pca.explained_variance_ratio_.cumsum())
            if x >= 0.9
        ]
    )
    + 1
)
if len(pca.explained_variance_ratio_.cumsum()) == 1:
    ncomp = 1
elif ncomp < 2:
    ncomp = 2

# Visualització Gràfica

# import matplotlib.pyplot as plt
# plt.style.use('seaborn')
# features = range(pca.n_components_)
# _, ax = plt.subplots()
# ax.plot(features, pca.explained_variance_ratio_.cumsum(), '-o', color='black')
# ax.set(title='Elbow plot',
#        xlabel='Component',
#        ylabel='Variance explained');
# plt.show()

# Estandaritzem variables i guardem la mitjana i la variança utilitzades.
scaler = StandardScaler()
standar = scaler.fit_transform(df)
means = scaler.mean_
variances = scaler.var_

# Transformem les observacions i guardem els eixos principals en l'espai de les variables
pca = PCA(n_components=ncomp)
pca.fit(standar)
components = pca.fit(standar).components_
standar_transformed = np.matmul(
    standar, components.T
)  # Equivalent a pca.transform(standar)


"""
Per ajustar el número de components del KMeans fem servir la prediction strength
en validació creuada: Partim el total d'observacions en 4 Folds, en cada iteració
s'utilitzen 3 folds que representen el training set i 1 fold que representa el 
test set. Ajustem Kmeans per separat al training set i al test set i calculem la 
prediction strength. En la següent iteració un fold diferent es representat com 
a test set i la resta de folds representen el training set. Un cop finalitzada 
la quarta iteració, calculem el promig de les prediction strength obtingudes. 
Repetim els passos per un rang de 1 a 10 components i seleccionem el màxim número
de components on el promig de prediction strengths calculades >= 0.8. 
"""
kf = KFold(n_splits=4)
mean_strengths = []
clusters = range(1, 10)
for k in clusters:
    strengths = []
    for train, test in kf.split(standar_transformed):
        X_train = standar_transformed[[train]]
        X_test = standar_transformed[[test]]
        model_train = KMeans(n_clusters=k, n_init=20, random_state=2).fit(X_train)
        model_test = KMeans(n_clusters=k, n_init=20, random_state=2).fit(X_test)
        pred_str = get_prediction_strength(
            k, model_train.cluster_centers_, X_test, model_test.labels_
        )
        strengths.append(pred_str)
    mean_strengths.append(np.mean(strengths))
n_clusters = max([index for index, x in enumerate(mean_strengths) if x >= 0.8]) + 1
n_clusters

# Visualització Gràfica

# _, ax = plt.subplots()
# ax.plot(clusters, mean_strengths, '-o', color='black')
# ax.axhline(y=0.8, c='red');
# ax.set(title='Determining the optimal number of clusters',
#        xlabel='number of clusters',
#        ylabel='prediction strength');
# plt.show()

# Ajustem K-Means amb n_clusters a les dades i guardem els centroides
model = KMeans(n_clusters=n_clusters, n_init=20, random_state=2)
model.fit(standar_transformed)
centroids = model.cluster_centers_

# Visualització 2 PCA

# import seaborn as sns
# ax = sns.scatterplot(x=standar_transformed[:, 0], y=standar_transformed[:, 1])
# ax = sns.scatterplot(x=standar_transformed[:, 0], y=standar_transformed[:, 1], hue=model.labels_)
# ax = sns.scatterplot(x=centroids[:,0],y=centroids[:,1],c='red',s=100)
# ax.set_title('Standarized 2 PCA', fontsize=16);
# plt.show()

# Exportem les dades del learner
with open("param.out", "w") as my_file:
    np.savetxt(my_file, [n_clusters], fmt="%i")
    np.savetxt(my_file, centroids, fmt="%.4f")
    np.savetxt(my_file, [means], fmt="%.4f")
    np.savetxt(my_file, [variances], fmt="%.2f")
    np.savetxt(my_file, components, fmt="%.4f")
print("Array exported to file")
