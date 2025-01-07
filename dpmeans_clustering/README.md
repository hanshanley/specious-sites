# Code for performing DP-Means Clustering via Cosine Similarity Metric

This code is a retrofitted version of DP-Means clustering as released by Dineri et al. https://github.com/BGU-CS-VIL/pdc-dp-means/tree/main/paper_code but instead utilizes cosine similarity. This version utilized for clustering sentence embeddings removes the random initialization step.

##  MiniBatch PDC-DP-Means via Cosine Similarity

In order to install this, you must clone scikit-learn from: `https://github.com/scikit-learn/scikit-learn.git`.

Note to use this code as is, you must use initial use following versions of these libraries:

```
scikit-learn>=1.2,<1.3
numpy>=1.23.0, <2
```

Navigate to the directory `sklearn/cluster` and replace the files `__init__.py`, `_k_means_lloyd.pyx` and `_kmeans.py` with the respective files in this directory.
Next, you need to install sklearn from source. To do so, follow the directions here: https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge.

To cluster using DP-Means, you can simply use `from sklearn.cluster import MiniBatchDPMeans, DPMeans`. In general, the parameters are the same as the `K-Means` counterpart:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
