This is the week 2 materials for course 1. Here list the main contents:
- [Neural Network Basics](#neural-network-basics)
 - [Logistic Regression as a Neural Network](#logistic-regression-as-a-neural-network)
   - [Binary classification](#binary-classification)
   - [Logistic Regression](#logistic-regresssion)
   - [Logistic Regression Cost Function](#logstic-regression-cost-function)
   - [Gradient Descent](#gradient-descent)
   - [Derivatives](#derivatives)
   - [Computation Graph](#computation-graph)
   - [Logistic Regression Gradient Descent](#logistic-regression-gradient-descent)
   - [Gradient Descent on m examples](#gradient-descent-on-m-examples)
 - [Python and Vectrization](#python-and-vectorization)
   - [Vectorization](#vectorization)
   - [Vectorizing logistic regression](vectorizng-logistic-regression)
   - [Broadcasting in Python](#broadcasting-in-python)
   - [Note on Python and Numpy](note-on-python-and-numpy)
# Neural Network Basics
> Learn to set up a machine learning problem with a neural network mindset. Learn to use vectorization to speed up your models.
## Logistic Regression as a Neural Network
### Binary Classification
Logistic Regression is an algrithm for binary classification. eg. input features of an image(x) to classify whether it's a cat (y=1) or not (y=0)

**Binary Classification**: the task of classifying the elements of a given set into two groups (predicting which group each one belongs to) on the basis of a classification rule.
   ! [](images/binary.png)
Some notation rules:
- M: number of examples, M_train/M_test.
- X: uppercase X, reprensent the input of the whole training set, X.shape=(NX,M).NX is the number of input's features, eg,the pixel of images.
- Y: uppercase Y, represent the output of the whole training set, Y.shape=(NY,M).NY is the number of output's class, eg, is a cat or not.
### Logistic Regression
Logistic Regression is a learning algrithm used in a supervised learning when the output Y is either 0 or 1 (binary classification)

Given parameters W and real number b, simple linear regression will calaculate y_hat=W_transpose*X+b (which can be any value). But, for logistic regression which only expect 0/1,a sigmoid function is applied.
y_hat= sigmoid(W_transpose+b)

**Sigmoid function**
![](images/sigmoid.png)
