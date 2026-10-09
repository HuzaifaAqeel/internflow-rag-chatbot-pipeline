# Regularization

Regularization is a family of techniques that discourage overly complex models in order to improve generalization. The core idea is to add a penalty for complexity to the training objective, so the optimizer must trade off fitting the training data against keeping the model simple.

L2 regularization (ridge) adds the sum of squared weights to the loss, shrinking all weights toward zero without removing any. L1 regularization (lasso) adds the sum of absolute weights, which drives some weights exactly to zero and therefore performs automatic feature selection. Elastic net combines both penalties. The regularization strength, usually called lambda or alpha, is tuned by cross-validation.

In neural networks the most popular regularizer is dropout: during training, each neuron is randomly deactivated with some probability, forcing the network to learn redundant representations that do not depend on any single neuron. Other neural regularizers include weight decay (L2 on the weights), batch normalization (which has a mild regularizing side effect), data augmentation, and early stopping — halting training when validation loss stops improving. The right amount of regularization is found empirically: too little and the model overfits, too much and it underfits.
