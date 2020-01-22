# Errata

## Chapter 1

- pg 10: 5th row from the bottom, x^{(i)} \epsilon \mathbb{R}^{150 x 1}, should be x_{j} \epsilon \mathbb{R}^{150 x 1}. (Was correct in the 2nd edition.)

## Chapter 2

- pg 42: It should be "eta=0.01" instead of "eta=0.1" in the sentence

> So, let's choose two different learning rates, eta = 0.1 and eta = 0.0001, to start with and plot the cost functions versus the number of epochs to see how well the Adaline implementation learns from the training data.


## Chapter 13

- pg. 469: Instead of `tf.keras.activations.tanh(z)` it should be `tf.keras.activations.relu(z)`.
