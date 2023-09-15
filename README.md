# Creating a clustering learner
Goal: Build an algorithm automatically for grouping similar results in an election dataset.

Create two scripts called *learner* and *predictor*:
  - The learner script should read an election dataset stored in a csv file called *training.csv*, and create a param.out file containing the hyperparameters and parameters that will be used by a clustering algorithm.
  - The predictor script should read an election dataset stored in a csv file called testing.csv and the param.out file and should run the clustering algorithm by returning a csv file called clustering.out with the cluster assignment for each row in the election dataset.

Approach taken:
First I compute the mahalanobis distance of every observation to detect outliers and delete them, because outliers have high influence in K-Means clustering.
Then I apply a dimensionality reduction (PCA) transformation to the data because K-Means scale bad with high dimensional data. After that I combine *K-Fold CV* and *Prediction strenght method* to select the optimum number of components for K-Means.
