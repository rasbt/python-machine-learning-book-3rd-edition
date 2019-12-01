Python Machine Learning - Code Examples


##  Chapter 18: Reinforcement Learning for Decision Making in Complex Environments


### Chapter Outline

- Introduction: learning from experience
  - Understanding reinforcement learning
  - Defining the agent-environment interface of a reinforcement learning system
  - The theoretical foundations of RL
    - Markov decision processes
    - The mathematical formulation of Markov decision processes
    - Visualization of a Markov process
    - Episodic versus continuing tasks
  - RL terminology: return, policy, and value function
    - The return
    - Policy
    - Value function
  - Dynamic programming using the Bellman equation
- Reinforcement learning algorithms
  - Dynamic programming
    - Policy evaluation – predicting the value function with dynamic programmin
    - Improving the policy using the estimated value function
    - Policy iteration
    - Value iteration
  - Reinforcement learning with Monte Carlo
    - State-value function estimation using MC
    - Action-value function estimation using MC
    - Finding an optimal policy using MC control
    - Policy improvement – computing the greedy policy from the action-value function
  - Temporal difference learning
    - TD prediction
    - On-policy TD control (SARSA)
    - Off-policy TD control (Q-learning)
- Implementing our first RL algorithm
  - Introducing the OpenAI Gym toolkit
    - Working with the existing environments in OpenAI Gym
  - A grid world example
    - Implementing the grid world environment in OpenAI Gym
  - Solving the grid world problem with Q-learning
    - Implementing the Q-learning algorithm
- A glance at deep Q-learning
  - Training a DQN model according to the Q-learning algorithm
    - Replay memory
    - Determining the target values for computing the loss
  - Implementing a deep Q-learning algorithm
- Chapter and book summary

### A note on using the code examples

The recommended way to interact with the code examples in this book is via Jupyter Notebook (the `.ipynb` files). Using Jupyter Notebook, you will be able to execute the code step by step and have all the resulting outputs (including plots and images) all in one convenient document.

![](../ch02/images/jupyter-example-1.png)



Setting up Jupyter Notebook is really easy: if you are using the Anaconda Python distribution, all you need to install jupyter notebook is to execute the following command in your terminal:

    conda install jupyter notebook

Then you can launch jupyter notebook by executing

    jupyter notebook

A window will open up in your browser, which you can then use to navigate to the target directory that contains the `.ipynb` file you wish to open.

**More installation and setup instructions can be found in the [README.md file of Chapter 1](../ch01/README.md)**.

**(Even if you decide not to install Jupyter Notebook, note that you can also view the notebook files on GitHub by simply clicking on them: [`ch18.ipynb`](ch18.ipynb))**

In addition to the code examples, I added a table of contents to each Jupyter notebook as well as section headers that are consistent with the content of the book. Also, I included the original images and figures in hope that these make it easier to navigate and work with the code interactively as you are reading the book.

![](../ch02/images/jupyter-example-2.png)


When I was creating these notebooks, I was hoping to make your reading (and coding) experience as convenient as possible! However, if you don't wish to use Jupyter Notebooks, I also converted these notebooks to regular Python script files (`.py` files) that can be viewed and edited in any plaintext editor. 
