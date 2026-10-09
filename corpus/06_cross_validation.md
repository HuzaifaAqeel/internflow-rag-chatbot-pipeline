# Cross-Validation

Cross-validation is a technique for estimating how a model will perform on unseen data while making efficient use of a limited dataset. The most common variant is k-fold cross-validation: the data is split into k equally sized folds, the model is trained on k-1 folds and evaluated on the remaining fold, and this is repeated so each fold serves once as the validation set. The k scores are averaged to produce a single robust estimate.

Stratified k-fold preserves the class proportions in each fold and should be used for classification problems, especially imbalanced ones. Leave-one-out cross-validation is the extreme case with k equal to the number of samples; it is unbiased but expensive. For time series data, standard k-fold is invalid because it leaks future information into training — instead use forward-chaining splits where each validation fold comes strictly after its training folds.

Cross-validation is also the standard way to tune hyperparameters: for each candidate configuration, compute the cross-validated score and pick the best. Nested cross-validation separates the tuning loop from the evaluation loop to avoid optimistic bias in the final performance estimate.
