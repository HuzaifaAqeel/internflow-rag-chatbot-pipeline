# Supervised Learning

Supervised learning is the most widely used machine learning paradigm. The algorithm learns a mapping from inputs to outputs using a training set of labelled examples. Each example consists of features (the input variables) and a label (the correct answer the model should predict).

Supervised problems split into two types. Classification predicts a discrete category: examples include spam detection, sentiment analysis, and image recognition. Common classification algorithms are logistic regression, decision trees, random forests, support vector machines, and neural networks. Regression predicts a continuous value, such as house prices or temperature; linear regression, gradient boosting, and neural networks are popular choices here.

Model quality is judged on a held-out test set that was not used during training. For classification we report accuracy, precision, recall, and F1-score; for regression we use mean squared error (MSE) or mean absolute error (MAE). A key danger is overfitting, where the model memorizes the training data but fails on new data. Techniques to fight overfitting include cross-validation, regularization, and collecting more training data.
