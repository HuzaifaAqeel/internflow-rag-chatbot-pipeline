# Overfitting and Underfitting

Overfitting and underfitting describe the two failure modes of model training. An underfit model is too simple: it has high error on both the training set and the test set because it cannot capture the underlying pattern — imagine fitting a straight line to curved data. An overfit model is too complex relative to the data: it achieves near-zero training error by memorizing noise, but its test error is high because the memorized noise does not repeat in new data.

The bias-variance tradeoff formalizes this tension. High bias (underfitting) means the model's assumptions are too rigid; high variance (overfitting) means small changes in the training data cause large changes in predictions. The sweet spot minimizes total error on unseen data.

Practical remedies for overfitting: gather more training data, simplify the model, add L1 or L2 regularization penalties, use dropout in neural networks, apply data augmentation, and stop training early when validation error starts rising. For underfitting: use a more expressive model, engineer better features, reduce regularization strength, and train longer. Cross-validation gives a reliable estimate of where a model sits on this spectrum.
