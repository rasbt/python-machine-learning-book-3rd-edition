# Errata

## Chapter 1

- pg 10: 5th row from the bottom, x^{(i)} \epsilon \mathbb{R}^{150 x 1}, should be x_{j} \epsilon \mathbb{R}^{150 x 1}. (Was correct in the 2nd edition.)

## Chapter 2

- pg 25: the superscripts (i) in $ \Delta w_{j}=\left(1^{(i)}-(-1)^{(i)}\right) 2^{(i)}=(2) 2^{(i)}=4 $ were initially used for illustration purposes in previous editions but are not necessary and could/should be removed

- pg. 42: It should be "eta=0.01" instead of "eta=0.1" in the sentence

> So, let's choose two different learning rates, eta = 0.1 and eta = 0.0001, to start with and plot the cost functions versus the number of epochs to see how well the Adaline implementation learns from the training data.

## Chapter 3

- pg 56: The comment about the `n_iter` method can be ignored as it does not apply to recent scikit-learn versions (0.21 and newer)

## Chapter 5

- pg. 162. In `S_i = \sum_{x in D_i} (x - m_i) (x - m_i)^{\top}`, the transpose should be on the first row vector, i.e., `S_i = \sum_{x in D_i} (x - m_i)^{\top} (x - m_i)`.

## Chapter 13

- pg. 469: Instead of `tf.keras.activations.tanh(z)` it should be `tf.keras.activations.relu(z)`.
