# Feature Engineering

Feature engineering is the craft of turning raw data into informative inputs for a model, and it often matters more than the choice of algorithm. Good features expose the signal; bad features bury it.

For numerical features, common transforms are scaling (standardization to zero mean and unit variance, or min-max normalization to a fixed range), log transforms for skewed distributions, and binning continuous values into categories. Tree-based models are scale-invariant, but distance-based models like k-NN and SVMs, and gradient-based neural networks, train much better on scaled inputs.

Categorical features need encoding. One-hot encoding creates a binary column per category and works well for low-cardinality variables. For high-cardinality variables (thousands of categories), target encoding replaces each category with the mean target value — powerful but prone to leakage if not done inside cross-validation folds. Embeddings learned by neural networks are the modern alternative for very large vocabularies.

Text and datetime features deserve special handling: extract year, month, day-of-week, and hour from timestamps; for text, TF-IDF or pretrained embeddings beat raw counts. Always fit transformers (scalers, encoders) on the training split only, then apply them to validation and test data — fitting on the full dataset leaks information and inflates scores.
