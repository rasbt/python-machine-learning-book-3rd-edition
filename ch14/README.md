Python Machine Learning - Code Examples


##  Chapter 14: Going Deeper – The Mechanics of TensorFlow


### Chapter Outline

- The key features of TensorFlow
  - TensorFlow's computation graphs: migrating to TensorFlow v2
     - Understanding computation graphs
     - Creating a graph in TensorFlow v1.x
     - Migrating a graph to TensorFlow v2
     - Loading input data into a model: TensorFlow v1.x style
     - Loading input data into a model: TensorFlow v2 style
  - Improving computational performance with function decorators
  - TensorFlow Variable objects for storing and updating model parameters
  - Computing gradients via automatic differentiation and GradientTape
     - Computing the gradients of the loss with respect to trainable variables
     - Computing gradients with respect to nontrainable tensors
     - Keeping resources for multiple gradient computations
- Simplifying implementations of common architectures via the Keras API
  - Solving an XOR classification problem
  - Making model building more flexible with Keras' functional API
  - Implementing models based on Keras' Model class
  - Writing custom Keras layers
- TensorFlow Estimators
  - Working with feature columns
  - Machine learning with pre-made Estimators
  - Using Estimators for MNIST handwritten digit classification
  - Creating a custom Estimator from an existing Keras model
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

**(Even if you decide not to install Jupyter Notebook, note that you can also view the notebook files on GitHub by simply clicking on them: [`ch14_part1.ipynb`](ch14_part1.ipynb) and [`ch14_part2.ipynb`](ch14_part2.ipynb))**

In addition to the code examples, I added a table of contents to each Jupyter notebook as well as section headers that are consistent with the content of the book. Also, I included the original images and figures in hope that these make it easier to navigate and work with the code interactively as you are reading the book.

![](../ch02/images/jupyter-example-2.png)


When I was creating these notebooks, I was hoping to make your reading (and coding) experience as convenient as possible! However, if you don't wish to use Jupyter Notebooks, I also converted these notebooks to regular Python script files (`.py` files) that can be viewed and edited in any plaintext editor. 
