# Gradient Descent

Gradient descent is the workhorse optimization algorithm behind most machine learning training. The idea is simple: to minimize a loss function, repeatedly take small steps in the direction of steepest descent, which is given by the negative gradient of the loss with respect to the model parameters.

The learning rate controls the step size and is the most important hyperparameter. If it is too large, the optimizer overshoots the minimum and may diverge; if it is too small, training takes an impractically long time. Learning rate schedules, such as step decay or cosine annealing, start large and shrink over time to combine fast early progress with stable final convergence.

Three classic variants differ in how much data each step uses. Batch gradient descent computes the gradient over the entire dataset — accurate but slow and memory-hungry. Stochastic gradient descent (SGD) uses a single sample per step — noisy but fast and able to escape shallow local minima. Mini-batch gradient descent strikes the balance, using batches of 32 to 256 samples, and is the default in deep learning. Modern adaptive optimizers like Adam, RMSprop, and Adagrad adjust the learning rate per parameter automatically and usually converge faster than plain SGD.
