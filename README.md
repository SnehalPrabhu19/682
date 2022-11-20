# Exploiting Loss Function Properties for Improved Performance

The training of a neural network is supervised by the
loss function, which acts as a learning objective with the
goal of improving the accuracy of a model. It does so by
managing the weights in neural networks in such a way
that the performance is maximized. Generally, individual
loss functions are used to train neural networks for object
recognition tasks. However, most other loss functions might
also have inherent attractive properties that would benefit
in training the model. In this research paper, we suggest a
combined loss function that exploits the advantages of multiple loss functions that have complimentary properties. We
do so by analyzing these properties for a set of functions
which are Hinge Loss, Softmax Loss, L-Softmax Loss and
A-Softmax Loss and discussing how each loss function benefits the overall training of a neural network. We support
this argument through an experimental analysis on an image classification task that employs a fully connected neural network on the CIFAR-10 dataset. We introduce various combinations of losses to train on this model and perform hyper parameter optimization through grid search to
assign weights to each loss function in the combined loss.
We evaluate the classification accuracies for each loss function and observe an improved performance in the combined
loss function as we predicted
