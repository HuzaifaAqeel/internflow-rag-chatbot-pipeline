# Neural Networks

A neural network is a model composed of layers of interconnected units called neurons. Each neuron computes a weighted sum of its inputs, adds a bias term, and passes the result through a non-linear activation function such as ReLU, sigmoid, or tanh. Stacking layers lets the network learn hierarchical representations: early layers detect simple patterns while deeper layers combine them into complex concepts.

Training uses backpropagation, which applies the chain rule of calculus to compute the gradient of a loss function with respect to every weight in the network. An optimizer — most commonly stochastic gradient descent (SGD) or Adam — then updates the weights in the direction that reduces the loss. This loop repeats over many epochs until the loss converges.

Key hyperparameters include the learning rate, batch size, number of layers, and number of neurons per layer. Too high a learning rate causes divergence; too low makes training painfully slow. Regularization methods such as dropout, weight decay, and early stopping help the network generalize instead of memorizing the training set.
