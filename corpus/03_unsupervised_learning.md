# Unsupervised Learning

Unsupervised learning finds structure in data that has no labels. Because there is no "correct answer" to check against, evaluation is harder and often relies on human judgment or downstream task performance.

Clustering is the best-known unsupervised task. K-means partitions data into k groups by alternating between assigning points to the nearest cluster centroid and recomputing the centroids; the number k must be chosen in advance, often with the elbow method. DBSCAN finds clusters of arbitrary shape based on point density and can label sparse points as noise. Hierarchical clustering builds a tree of nested clusters that can be cut at any level.

Dimensionality reduction is another major task. Principal Component Analysis (PCA) projects data onto the directions of maximum variance, which is useful for visualization and for speeding up downstream models. Autoencoders are neural networks that learn a compressed representation by reconstructing their own input.

Common applications include customer segmentation, anomaly detection in fraud or manufacturing, topic modelling of documents, and recommendation systems that group similar users or items.
