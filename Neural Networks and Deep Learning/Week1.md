This is the first course of the specialzation, required 4 weeks' study, here list the week 1's main content:

- [Introduction to Deep Learning](#introduction_to_deep_learning)
   - [What is a Neural Network](#what_is_a_neural_network)
   - [Supervised Learning with Neural Networks]
   - [Why is Deep Learning taking off]

## Introduction to Deep Learning
Learning objective: 
> Understand the major trends driving the rise of deep learning.
> Be able to explain how deep learning is applied to supervised learning.
> Understand what are the major categories of models (such as CNNs and RNNs), and when they should be applied.
> Be able to recognize the basics of when deep learning will (or will not) work well.
### What is a Neural Network?
From lecture (a more figurative explanation through example):

A simple neural network: predict house price (Y) with house size (X). The circle which represents a single **neuron** in the neural network implements the predict function.
![](images/simple_nn.png)

A larger neural network: predict house price (Y) with house size(x1), # of bedrooms(x2),zipcode(x3) and wealth(x4). Instead of predict by a single neuron, there is a hidden layer with multiple neurons to process the raw inputs.
![](images/larger_nn.png)

From other references (a more conceptive explanation):

> A neural network is a series of algorithms that endeavors to recognize **underlying relationships in a set of data** through a process that mimics the way the human brain operates. Neural networks can **adapt to changing input**; so the network generates the best possible result without needing to redesign the output criteria.

Terminologies to take away:

**RELU**(Rectified Linear Unit), is a type of activation function. Mathematically, it is defined as **y = max(0, x)**. Visually, it looks like the following: 
![](images/relu.png)

ReLU is the most commonly used activation function in neural networks, especially in CNNs.

**Activation function**.In artificial neural networks, the activation function of a node defines the output of that node given an input or set of inputs. With it, NN can learn and make sense of something really **complicated and Non-linear complex functional mappings** between the inputs and response variable.They introduce **non-linear properties** to our Network.

