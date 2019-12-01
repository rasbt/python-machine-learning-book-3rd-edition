Python Machine Learning - Code Examples


##  Chapter 13: Parallelizing Neural Network Training with TensorFlow 


### Chapter Outline

- TensorFlow and training performance
  - Performance challenges
  - What is TensorFlow?
  - How we will learn TensorFlow
- First steps with TensorFlow
  - Installing TensorFlow
  - Creating tensors in TensorFlow
  - Manipulating the data type and shape of a tensor
  - Applying mathematical operations to tensors
  - Split, stack, and concatenate tensors
  - Building input pipelines using tf.data – the TensorFlow Dataset API
    - Creating a TensorFlow Dataset from existing tensors
    - Combining two tensors into a joint dataset
    - Shuffle, batch, and repeat
    - Creating a dataset from files on your local storage disk
    - Fetching available datasets from the `tensorflow_datasets` library
- Building an NN model in TensorFlow
  - The TensorFlow Keras API (tf.keras)
  - Building a linear regression model
  - Model training via the `.compile()` and `.fit()` methods 
  - Building a multilayer perceptron for classifying flowers in the Iris dataset
  - Evaluating the trained model on the test dataset
  - Saving and reloading the trained model
- Choosing activation functions for multilayer NNs
  - Logistic function recap
  - Estimating class probabilities in multiclass classification via the softmax function
  - Broadening the output spectrum using a hyperbolic tangent
  - Rectified linear unit activation
- Summary

### A note on using the code examples

The recommended way to interact with the code examples in this book is via Jupyter Notebook (the `.ipynb` files). Using Jupyter Notebook, you will be able to execute the code step by step and have all the resulting outputs (including plots and images) all in one convenient document.

![](../ch02/images/jupyter-example-1.png)



Setting up Jupyter Notebook is really easy: if you are using the Anaconda Python distribution, all you need to install jupyter notebook is to execute the following command in your terminal:

    conda install jupyter notebook

Then you can launch jupyter notebook by executing

    jupyter notebook

A window will open up in your browser, which you can then use to navigate to the target directory that contains the `.ipynb` file you wish to open.

**More installation and setup instructions can be found in the [README.md file of Chapter 1](../ch01/README.md)**.

**(Even if you decide not to install Jupyter Notebook, note that you can also view the notebook files on GitHub by simply clicking on them: [`ch13_part1.ipynb`](ch13_part1.ipynb) and [`ch13_part2.ipynb`](ch13_part2.ipynb))**

In addition to the code examples, I added a table of contents to each Jupyter notebook as well as section headers that are consistent with the content of the book. Also, I included the original images and figures in hope that these make it easier to navigate and work with the code interactively as you are reading the book.

![](../ch02/images/jupyter-example-2.png)


When I was creating these notebooks, I was hoping to make your reading (and coding) experience as convenient as possible! However, if you don't wish to use Jupyter Notebooks, I also converted these notebooks to regular Python script files (`.py` files) that can be viewed and edited in any plaintext editor. 
