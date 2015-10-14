__author__ = 'T7'


# coding: utf-8

import random
import numpy as np
from cs224d.data_utils import *
import matplotlib.pyplot as plt


def softmax(x):
    """ Softmax function """
    ###################################################################
    # Compute the softmax function for the input here.                #
    # It is crucial that this function is optimized for speed because #
    # it will be used frequently in later code.                       #
    # You might find numpy functions np.exp, np.sum, np.reshape,      #
    # np.max, and numpy broadcasting useful for this task. (numpy     #
    # broadcasting documentation:                                     #
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)  #
    # You should also make sure that your code works for one          #
    # dimensional inputs (treat the vector as a row), you might find  #
    # it helpful for your later problems.                             #
    ###################################################################

    ### YOUR CODE HERE
    x = x.T  # avoid using np.reshape each time
    x -= np.amax(x, axis=0)
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=0)
    x = x_exp/x_sum
    x = x.T
    ### END YOUR CODE
    return x

print softmax(np.array([[1,2],[3,4]]))

# Verify your softmax implementation

print "=== For autograder ==="
print softmax(np.array([[1001,1002],[3,4]]))
print softmax(np.array([[-1001,-1002]]))

# ## 2. Neural network basics
#
# *Please answer the second complementary question before starting this part.*
#
# In this part, you're going to implement
#
# * A sigmoid activation function and its gradient
# * A forward propagation for a simple neural network with cross-entropy cost
# * A backward propagation algorithm to compute gradients for the parameters
# * Gradient / derivative check

# In[5]:

def sigmoid(x):
    """ Sigmoid function """
    ###################################################################
    # Compute the sigmoid function for the input here.                #
    ###################################################################

    return 1/(1 + np.exp(-x))

def sigmoid_grad(f):
    """ Sigmoid gradient function """
    ###################################################################
    # Compute the gradient for the sigmoid function here. Note that   #
    # for this implementation, the input f should be the sigmoid      #
    # function value of your original input x.                        #
    ###################################################################

    return f*(1-f)

# Now, use the functions you just implemented, fill in the following functions to implement a neural network with one sigmoid hidden layer. You might find the handout and your answers to the second complementary problem helpful for this part.

# In[7]:

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """
    Gradient check for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### YOUR CODE HERE: try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it
        ### possible to test cost functions with built in randomness later

        x[ix] += h # increment by h
        fxph, _ = f(x) # evaluate f(x + h)
        x[ix] -= 2 * h # increment by h
        fxmh, _ = f(x) # evaluate f(x - h)
        x[ix] += h # reset

        numgrad = (fxph - fxmh) / (2 * h)

        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"


# In[8]:

# Sanity check for the gradient checker
quad = lambda x: (np.sum(x ** 2), x * 2)

print "=== For autograder ==="
# gradcheck_naive(quad, np.array(123.456))      # scalar test
# gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
# gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test


# In[9]:

# Set up fake data and parameters for the neural network
"""
N = 20
dimensions = [10, 5, 10]
data = np.random.randn(N, dimensions[0])   # each row will be a datum
labels = np.zeros((N, dimensions[2]))
for i in xrange(N):
    labels[i,random.randint(0,dimensions[2]-1)] = 1

params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )


# In[10]:

def forward_backward_prop(data, labels, params):
   # Forward and backward propagation for a two-layer sigmoidal network
    ###################################################################
    # Compute the forward propagation and for the cross entropy cost, #
    # and backward propagation for the gradients for all parameters.  #
    ###################################################################

    ### Unpack network parameters (do not modify)
    t = 0
    W1 = np.reshape(params[t:t+dimensions[0]*dimensions[1]], (dimensions[0], dimensions[1]))
    t += dimensions[0]*dimensions[1]
    b1 = np.reshape(params[t:t+dimensions[1]], (1, dimensions[1]))
    t += dimensions[1]
    W2 = np.reshape(params[t:t+dimensions[1]*dimensions[2]], (dimensions[1], dimensions[2]))
    t += dimensions[1]*dimensions[2]
    b2 = np.reshape(params[t:t+dimensions[2]], (1, dimensions[2]))

    ### YOUR CODE HERE: forward propagation
    # cross entropy cost equals to the cost function of softmax
    # score = softmax(sigmoid(sigmoid(data.dot(W1) + b1).dot(W2) + b2))
    # because the activations and its input is used in calculating propagation, step by step
    z1 = data.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    # a2 = sigmoid(z2)
    score = softmax(z2)
    # cost function only cares the score of true labels
    cost = -np.sum(np.log(score*labels))
    # temporally without regularization !
    cost = cost/data.shape[0]
    # cost = ...

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    error_last = (score - labels) / data.shape[0]
    gradb2 = np.sum(error_last, axis=0)
    gradW2 = a1.T.dot(error_last)
    error_hidden = error_last.dot(W2.T) * sigmoid_grad(z1)
    gradb1 = np.sum(error_hidden, axis=0)
    gradW1 = data.T.dot(error_hidden)
    #gradW1 = ...
    #gradb1 = ...
    #gradW2 = ...
    #gradb2 = ...

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))

    return cost, grad


# In[29]:

# Perform gradcheck on your neural network
print "=== For autograder ==="
gradcheck_naive(lambda params: forward_backward_prop(data, labels, params), params)

"""
# ## 3. Word2vec
#
# *Please answer the third complementary problem before starting this part.*
#
# In this part you will implement the `word2vec` models and train your own word vectors with stochastic gradient descent (SGD).

# In[16]:

# Implement your skip-gram and CBOW models here

# Interface to the dataset for negative sampling
dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)
def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return tokens[random.randint(0,4)], [tokens[random.randint(0, 4)] for i in xrange(2*C)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext


def softmaxCostAndGradient(predicted, target, outputVectors):
    """ Softmax cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, assuming the softmax prediction function and cross      #
    # entropy loss.                                                   #
    # Inputs:                                                         #
    #   - predicted: numpy ndarray, predicted word vector (\hat{r} in #
    #           the written component)                                #
    #   - target: integer, the index of the target word               #
    #   - outputVectors: "output" vectors for all tokens              #
    # Outputs:                                                        #
    #   - cost: cross entropy cost for the softmax word prediction    #
    #   - gradPred: the gradient with respect to the predicted word   #
    #           vector                                                #
    #   - grad: the gradient with respect to all the other word       #
    #           vectors                                               #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    # predicted 1 x d vector -> let it be (d, )
    # outputVectors |V| x d matrix
    wr = outputVectors.dot(predicted)
    wr_exp= np.exp(wr)
    wr_sum = np.sum(wr_exp)
    prob = wr_exp * 1.0 / wr_sum # (|V| , )

    cost = - wr[target] + np.log(wr_sum)

    gradPred = - outputVectors[target, :] + outputVectors.T.dot(prob)

    # grad = np.sum(prob) * np.ones([prob.shape[0], 1]).dot(predicted.reshape(1, predicted.shape[0]))
    grad = prob.reshape(prob.shape[0], 1).dot(predicted.reshape(1, predicted.shape[0]))

    grad[target, :] -= predicted

    ### END YOUR CODE

    return cost, grad #Pred#,  grad

def negSamplingCostAndGradient(predicted, target, outputVectors, K=10):
    """ Negative sampling cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, using the negative sampling technique. K is the sample  #
    # size. You might want to use dataset.sampleTokenIdx() to sample  #
    # a random word index.                                            #
    # Input/Output Specifications: same as softmaxCostAndGradient     #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE

    random.seed(1) # only used when gradient checking !
    neg_index = []  # use list to index ndarray !
    while len(neg_index) < K:
        i_rand = dataset.sampleTokenIdx()
        if i_rand != target:
            neg_index.append(i_rand)


    # neg_index = [0,2,0,2]
    # predicted 1 x d vector -> let it be (d, )
    # outputVectors |V| x d matrix
    targetVector = outputVectors[target, :]
    targetSigmoid = sigmoid(np.dot(targetVector, predicted))
    # target_prod = sigmoid(outputVectors[target, :].dot(predicted))  # number
    # negative vectors |K| x d matrix
    negVectors = outputVectors[neg_index, :]  # |K| x d
    negSigmoid = sigmoid(-np.dot(negVectors, predicted))  # (|K|, )

    cost = -np.log(targetSigmoid) - np.sum(np.log(negSigmoid))

    gradPred = -(1.0-targetSigmoid) * targetVector + negVectors.T.dot((1-negSigmoid))
    # gradPred = -gradPred

    grad = np.zeros(outputVectors.shape)

    for i in range(len(neg_index)):
        # for index in neg_index:
        # grad[index, :] += (1-negative_prod[index]) * predicted
        grad[neg_index[i], :] += (1.0-negSigmoid[i]) * predicted

    grad[target, ] = (targetSigmoid - 1.0) * predicted

    # END YOUR CODE

    return cost, grad# Pred#, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    ###################################################################
    # Implement the skip-gram model in this function.                 #
    # Inputs:                                                         #
    #   - currrentWord: a string of the current center word           #
    #   - C: integer, context size                                    #
    #   - contextWords: list of no more than 2*C strings, the context #
    #             words                                               #
    #   - tokens: a dictionary that maps words to their indices in    #
    #             the word vector list                                #
    #   - inputVectors: "input" word vectors for all tokens           #
    #   - outputVectors: "output" word vectors for all tokens         #
    #   - word2vecCostAndGradient: the cost and gradient function for #
    #             a prediction vector given the target word vectors,  #
    #             could be one of the two cost functions you          #
    #             implemented above                                   #
    # Outputs:                                                        #
    #   - cost: the cost function value for the skip-gram model       #
    #   - grad: the gradient with respect to the word vectors         #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    predicted_index = tokens[currentWord]
    predicted = inputVectors[predicted_index, :]
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    for i in range(2*C):
        target = tokens[contextWords[i]]
        c, gin, gout = word2vecCostAndGradient(predicted, target, outputVectors)
        cost += c
        gradIn[predicted_index, :] += gin
        # gradIn += gin
        gradOut += gout


    ### END YOUR CODE

    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """
    ###################################################################
    # Implement the continuous bag-of-words model in this function.   #
    # Input/Output specifications: same as the skip-gram model        #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    r = np.zeros(inputVectors.shape[1])
    context_index = [tokens[w] for w in contextWords]

    for w in contextWords:
        r += inputVectors[tokens[w]]

    gradIn = np.zeros(inputVectors.shape)
    c, gin, gout = word2vecCostAndGradient(r, tokens[currentWord], outputVectors)
    gradIn[context_index, :] = gin

    ### END YOUR CODE

    return c, gradIn, gout


# In[13]:

# Implement a function that normalizes each row of a matrix to have unit length
def normalizeRows(x):
    """ Row normalization function """

    ### YOUR CODE HERE
    square_sum_row = np.sqrt(np.sum(x ** 2, axis=1)).reshape(x.shape[0], 1)
    x /= square_sum_row
    ### END YOUR CODE

    return x

# Test this function
print "=== For autograder ==="
print normalizeRows(np.array([[3.0,4.0],[1, 2]]))  # the result should be [[0.6, 0.8], [0.4472, 0.8944]]


# In[17]:

# Gradient check!

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad

random.seed(31415)
np.random.seed(9265)
dummy_vectors = normalizeRows(np.random.randn(10,3))
dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])


index = np.random.randint(0, 5)
predicted = dummy_vectors[np.random.randint(0,5),:]
outputVectors = dummy_vectors[5:, :]
# print "==== Gradient check for  softmaxCostAndGradient ===="
# gradcheck_naive(lambda vec: softmaxCostAndGradient(vec, index, outputVectors), predicted)
# gradcheck_naive(lambda vec: softmaxCostAndGradient(predicted, index, vec), outputVectors)


print "==== Gradient check for  negSamplingCostAndGradient ===="
# gradcheck_naive(lambda vec: negSamplingCostAndGradient(vec, index, outputVectors), predicted)
gradcheck_naive(lambda vec: negSamplingCostAndGradient(predicted, index, vec), outputVectors)



"""
print "==== Gradient check for skip-gram ===="
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

print "\n==== Gradient check for CBOW      ===="
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
"""

"""
print "\n=== For autograder ==="
print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)
print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)
"""
