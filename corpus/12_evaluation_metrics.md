# Evaluation Metrics for Classification

Choosing the right metric is as important as choosing the right model, because the metric defines what "good" means. Accuracy — the fraction of correct predictions — is intuitive but misleading on imbalanced data: a model that always predicts the majority class can score 95% accuracy while being useless.

Precision and recall give a fuller picture for binary problems. Precision answers "of the items flagged positive, how many truly are?" while recall answers "of all truly positive items, how many did we catch?" The F1-score is their harmonic mean and is the standard single number when both matter. For multi-class problems these are averaged per class, either macro (each class weighted equally) or weighted (weighted by class frequency).

Beyond a single threshold, the ROC curve plots true-positive rate against false-positive rate across thresholds, and the area under it (ROC-AUC) summarizes ranking quality. For probabilistic models, log loss (cross-entropy) penalizes confident wrong predictions. Always report metrics on a held-out test set, and when comparing models use the same splits — otherwise the comparison is meaningless.
