This is the 3rd week of first course, in this week, the learning objective is to : Learn to build a neural network with one hidden layer, using forward propagation and backpropagation. 
The main content of this week as follows:
- [Neural Networks Overview](#neural-networks-overview)
- [Neural Network Representation](#neural-network-representation)
- [Computing a Neural Network's output](#computing-a-neural-network-output)
- [Vectorizing across multiple examples](#vectorizing-across-multiple-examples)
- [Activation Function](#activation-function)
- [Why do we need non-linear activation functions](#why-do-we-need-non-linear-activation-function)
- [Derivatives of activation functions](#derivatives-of-activation-functions)
- [Gradient Descent for Neural Networks](#gradient-descent-for-neural-networks)
- [Backpropagation Intuition](#backpropagation-intuition)
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
## Computing a neural network output
In this part, we only care about single traning example.

Notation convention: `a[l]j` refers to the jth node of lth layer

![](images/step.png)

The top part calculate each node in hidden layer, while bottom part vectorizing x,W,b,Z,A.
```
x is the input, with shape(Nx,1)

W1 is the weight parameter of L1, with shape(NL1,NL0) (Nl refers to the # of node in layer l)

b1 is the bias parameter of L1, with shape(NL1,1)

Z1 is the linear output of L1, with shape(NL1,1)

A1 is the final output (after activation) of L1, with shape(NL1,1)

W2 is the weight parameter of L2, with shape(NL2,NL1)

b2 is the bias parameter of L2, with shape(NL2,1)

Z2 is the linear output of L2, with shape(NL2,1)

A2 is the final output of L2, with shape(NL2,1)
```

This rule can be generalized to 
```
W[l] with shape(Nl,N[l-1])

b[l] with shape(Nl,1)

Z[l] with shape(Nl,1)

A[l] with shape(Nl,1)
```

The output computed with vectorizing `y_hat=A2=sigmoid(Z2),Z2=W2*A1+b2,A1=sigmoid(Z1),Z1=W1*x+1`
## Vectorizing across multiple examples
In this part, take mutiple training examples into consideration. m refers to the number of traning examples.

notation convention: `a[l](i) refers to ith traning example in layer l`

![](images/multiple.png)

The step by step calculation can be implemented with a for-loop:
```
for i=1 to m:
    Z[1](i)=W[1]*X[:,i]+b[1]
    A[1](i)=sigmoid(Z[1](i))
    Z[2](i)=W[2]*A[1](i)+b[2]
    A[2](i)=sigmoid(Z[2](i))
```
Vectoring X,Z and A:

```
X is the input, the whole training set, with shape (Nx,m)

Z[l] is the linear output of layer l, with shape(Nl,m)

A[l] is the final output of layer l ,with shape(Nl,m)
```
The output computed with vectorizing `y_hat=A2=sigmoid(Z2),Z2=W2*A1+b2,A1=sigmoid(Z1),Z1=W1*X+b1`
## Activation Function
![](images/activation.png)

| Function    | Pros | Cons|
| ----------- | ----------- | ----------- |
| Sigmoid      | useful in binary output | slow gradient descent when z is small/ large (slope close to 0) |
| tanh   | mean of output close to 0, centre data for next layer  | slow gradient descent when z is small/ large |
| RELU   | fast gradient descent as slope is linear when z is positive | slope close to 0 when z is negative |
|Leaky RELU | small slope (not 0) when z is negative | another learnable parameter to define -- slope when z is negative |

Principles of how to choose activation functions:

- Use sigmoid for binary classification (0/1), others Relu.
- Find the one better fit with your problem through try-out.
## Why do we need non linear activation function
If there is no activation function or alternatively, linear activation function, then no matterhow many hidden layers, the output will be linear and thus cannot handle more complex problems like logistic regression.

Linear activation function might be used in one place -- the output layer of regerssion problem.

## Derivatives of activation function
Derivative of sigmoid function
```
g(z)=1/(1+np.exp(-z))

g'(z)=g(z)*(1-g(z))
```
Derivative of tanh function
```
g(z)=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

g'(z)=1-g(z)^2
```
Derivative of Relu
```
g(z)=max(0,z)=0 if z<0,z if z>=0

g'(z)=0 if z<0, 1 if z>=0
```
Derivative of Leaky Relu
```
g(z)=max(0.01z,z)

g'(z)=0.01 if z<0, 1 if z>=0
```
## Gradient descent for neural networks
shallow NN with 2 layers

cost function:`J=(1/m)*sum(L(y[i],y_hat[i]))`

parameters:
```
W1 is the weight parameter of L1, with shape(NL1,NL0) (Nl refers to the # of node in layer l)

b1 is the bias parameter of L1, with shape(NL1,1)

W2 is the weight parameter of L2, with shape(NL2,NL1)

b2 is the bias parameter of L2, with shape(NL2,1)
```
Gradient descent for NN
```
Repeat {
        compute prediction y_hat[i] for i=1...m
        
        compute partial derivatives: dw[1],db[1],dw[2],db[2]    # details in below
        
        apply gradient descent:
        
        w[1]:= w[1]-learning_rate*dw[1]
        
        b[1]:= b[1]-learning_rate*db[1]
        
        w[2]:= w[2]-learning_rate*dw[2]
        
        b[2]:=b[2]-learning_rate*db[2]
```
Formulas for computing derivatives
![](images/formulas.png)
## Backpropagation intuition 
How do we calculate the derivatives (the backpropagation process)

![](images/intuition.png)

Vectorizing to m examples

![](images/intuition2.png)
## Random initialization
Why not initialize parameters with zeros (initialize bias terms b to 0 is ok, but initialize w to 0 is a problem):
- all hidden units in a certain layer are the same (computing the same function)
- after every single iteration, the hidden units are updated the same way, thus still computing the same function.

How to randomly initializing parameters:
```
W[1]=np.random((NL1,NL0))*0.01

b[1]=np.zeros((NL1,1))

w[2]=np.random((NL2,NL1))*0.01

b[1]=np.zeros((NL2,1))
```
Why multiply 0.01 with the random initialized w?

To make W small enough, the Z=W*X+b will be smaller, the A=g(z) is smaller, if g is sigmoid, g'(z) is bigger, thus make the training quiker.
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


