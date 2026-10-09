# Decision Trees and Random Forests

A decision tree is an interpretable model that makes predictions by asking a sequence of yes-or-no questions about the input features. Each internal node tests one feature (for example, "is age > 30?"), each branch follows an answer, and each leaf holds a prediction. Trees are grown by recursively choosing the split that best separates the data, measured by impurity criteria such as Gini impurity or entropy for classification and variance reduction for regression.

Single trees overfit easily: grown deep enough, a tree can memorize the training set. Two cures exist. Pruning removes branches that do not improve validation performance. Ensembling is more powerful: a random forest trains hundreds of trees on bootstrap samples of the data, each tree seeing only a random subset of features at each split, and averages their votes. This bagging procedure dramatically reduces variance while keeping the low bias of deep trees.

Gradient boosting takes a different ensemble route: trees are added sequentially, each new tree correcting the residual errors of the previous ones. Implementations like XGBoost, LightGBM, and CatBoost dominate tabular-data competitions. Feature importance scores from tree ensembles are a popular, model-native way to explain which inputs drive predictions.
