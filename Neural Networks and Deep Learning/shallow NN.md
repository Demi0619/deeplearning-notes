This is the 3rd week of first course, in this week, the learning objective is to : Learn to build a neural network with one hidden layer, using forward propagation and backpropagation. 
The main content of this week as follows:
- [Neural Networks Overview](#neural-networks-overview)
- [Neural Network Representation](#neural-network-representation)
- [Computing a Neural Network's output](#computing-a-neural-network's-output)
- [Vectorizing across multiple examples](#vectorizing-across-multiple-examples)
- [Explanation for Vectorized Implementation](#explanation-for-vectorized-implementation)
- [Activation Function](#activation-function)
- [Why do we need non-linear activation functions](#why-do-we-need-non-linear-activation-function)
- [Derivatives of activation functions](#derivatives-of-activation-functions)
- [Gradient Descent for Neural Networks](#gradient-descent-for-neural-networks)
- [Backpropagation Intuition](#backpropagation-ntuition)
- [Random Initialization](#random-initialization)
## Neural Networks overview
From previous study, a simple logistic regression can be represented as:

`Z=W*X+b A=sigmoid(Z)`

![](images/logstic.png)

A Neural Network with 1 hidden layer can be represented as (repeat logistic regression once):

`Z[1]=W[1]*X+b[1] A[1]=sigmoid(Z[1]) Z[2]=W[2]*A[1]+b[2] A[2]=sigmoid(Z[2])`

![](images/1nn.png)
## Neural Network Representation
A NN can be divided by 3 different layers: **Input Layer**(X);**Output Layer**(Y); **Hidden Layer**(layers in bettwen, not visible)

`# layers NN` '#' is defined by `# of hidden layers + output layers` (Input layer not counted)
## Computing a neural network's output
Step by step calculation:

Notation convention: `a[i]j` refers to the jth node of ith layer

![](images/step.png)

The top part calculate each node in hidden layer, while bottom part vectorizing X,W,b,Z,A.
`X is the input, with shape(Nx,m)

W1 is the weight parameter of L1, with shape(NL1,NL0) (Nl refers to the # of node in layer l)

b1 is the bias parameter of L1, with shape(NL1,1)

Z1 is the linear output of L1, with shape(NL1,m)

A1 is the final output (after activation) of L1, with shape(NL1,m)

W2 is the weight parameter of L2, with shape(NL2,NL1)

b2 is the bias parameter of L2, with shape(NL2,1)

Z2 is the linear output of L2, with shape(NL2,m)

A2 is the final output of L2, with shape(NL2,m)`

This rule can be generalized to 

`W[l] with shape(Nl,N[l-1])

b[l] with shape(Nl,1)

Z[l] with shape(Nl,m)

A[l] with shape(Nl,m)`

The output computed with vectorizing `y_hat=A2=sigmoid(Z2),Z2=W2*A1+b2,A1=sigmoid(Z1),Z1=W1*X+b1`


