Python Machine Learning - Code Examples


##  Chapter 16: Modeling Sequential Data Using Recurrent Neural Networks


### Chapter Outline

- Introducing sequential data
  - Modeling sequential data—order matters
  - Representing sequences
  - The different categories of sequence modeling
- RNNs for modeling sequences
  - Understanding the RNN looping mechanism
  - Computing activations in an RNN
  - Hidden-recurrence versus output-recurrence
  - The challenges of learning long-range interactions
  - Long short-term memory cells
- Implementing RNNs for sequence modeling in TensorFlow
  - Project one: predicting the sentiment of IMDb movie reviews
    - Preparing the movie review data
    - Embedding layers for sentence encoding
    - Building an RNN model
    - Building an RNN model for the sentiment analysis task
  - Project two: character-level language modeling in TensorFlow
    - Preprocessing the dataset
    - Building a character-level RNN model
    - Evaluation phase: generating new text passages
- Understanding language with the Transformer model
  - Understanding the self-attention mechanism
  - A basic version of self-attention
  - Parameterizing the self-attention mechanism with query, key, and value weights
  - Multi-head attention and the Transformer block
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

**(Even if you decide not to install Jupyter Notebook, note that you can also view the notebook files on GitHub by simply clicking on them: [`ch16_part1.ipynb`](ch16_part1.ipynb) and [`ch16_part2.ipynb`](ch16_part2.ipynb))**

In addition to the code examples, I added a table of contents to each Jupyter notebook as well as section headers that are consistent with the content of the book. Also, I included the original images and figures in hope that these make it easier to navigate and work with the code interactively as you are reading the book.

![](../ch02/images/jupyter-example-2.png)


When I was creating these notebooks, I was hoping to make your reading (and coding) experience as convenient as possible! However, if you don't wish to use Jupyter Notebooks, I also converted these notebooks to regular Python script files (`.py` files) that can be viewed and edited in any plaintext editor. 
