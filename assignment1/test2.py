__author__ = 'T7'
### Gradient check script !

import numpy as np
import random

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

def sigmoid(x):
    """ Sigmoid function """
    ###################################################################
    # Compute the sigmoid function for the input here.                #
    ###################################################################

    return 1/(1 + np.exp(-x))

def normalizeRows(x):
    """ Row normalization function """

    ### YOUR CODE HERE
    square_sum_row = np.sqrt(np.sum(x ** 2, axis=1)).reshape(x.shape[0], 1)
    x /= square_sum_row
    ### END YOUR CODE

    return x


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
    wr_exp = np.exp(wr)
    wr_sum = np.sum(wr_exp)
    prob = wr_exp * 1.0 / wr_sum  # (|V| , )

    cost = - wr[target] + np.log(wr_sum)

    gradPred = - outputVectors[target, :] + outputVectors.T.dot(prob)

    # grad = np.sum(prob) * np.ones([prob.shape[0], 1]).dot(predicted.reshape(1, predicted.shape[0]))
    grad = prob.reshape(prob.shape[0], 1).dot(predicted.reshape(1, predicted.shape[0]))

    grad[target, :] -= predicted

    ### END YOUR CODE

    return cost, gradPred,  grad


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

    grad = np.zeros(outputVectors.shape)

    for i in range(len(neg_index)):
        grad[neg_index[i], :] += (1.0-negSigmoid[i]) * predicted

    grad[target, ] = (targetSigmoid - 1.0) * predicted

    # END YOUR CODE

    return cost, gradPred, grad


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
    # predicted vector, the center vector, is in the inputVectors in skip-gram
    predicted = inputVectors[predicted_index, :]
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    C = len(contextWords)

    for i in range(C):
        # target the index of output vector, expected vector
        # the size of context words is at most 2*C
        target = tokens[contextWords[i]]
        c, gradPred, gOut = word2vecCostAndGradient(predicted, target, outputVectors)
        cost += c
        gradIn[predicted_index, :] += gradPred
        # gradIn += gin
        gradOut += gOut

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
    # predicted vector is the sum of contextWords vectors
    # context_index = [tokens[w] for w in contextWords]
    # predicted = np.sum(inputVectors[context_index, :], axis=0)

    target = tokens[currentWord]

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    C = len(contextWords)
    for i in range(C):
        predicted_index = tokens[contextWords[i]]
        predicted = inputVectors[predicted_index, :]
        c, gradPred, gout = word2vecCostAndGradient(predicted, target, outputVectors)

        cost += c
        gradIn[predicted_index, :] += gradPred
        # gradOut[target, :] += gout[target, :]
        gradOut += gout

    ### END YOUR CODE

    return cost, gradIn, gradOut


def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2, :]
    outputVectors = wordVectors[N/2:, :]

    random.seed(batchsize) # only used in gradient check !!!
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

"""
wordVectors = dummy_vectors
cost = 0.0
grad = np.zeros(wordVectors.shape)
N = wordVectors.shape[0]
inputVectors = wordVectors[:N/2, :]
outputVectors = wordVectors[N/2:, :]

C1 = random.randint(1, 5)
centerword, context = dataset.getRandomContext(C1)

print centerword, context
"""

# c, gin = skipgram(centerword, C1, context, dummy_tokens, inputVectors, outputVectors)

print "==== Gradient check for skip-gram ===="
# gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
# gradient for inputVectors  passed !
# gradcheck_naive(lambda vec: skipgram(centerword, C1, context, dummy_tokens, vec, outputVectors), inputVectors)
# gradient for outputVectors passed !
# gradcheck_naive(lambda vec: skipgram(centerword, C1, context, dummy_tokens, inputVectors, vec), outputVectors)

print "==== Gradient check for CBOW ===="
# gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
# gradient for inputVectors  not passed ! 8.6.2015
# gradcheck_naive(lambda vec: cbow(centerword, C1, context, dummy_tokens, vec, outputVectors), inputVectors)
# gradient for outputVectors passed !
# gradcheck_naive(lambda vec: cbow(centerword, C1, context, dummy_tokens, inputVectors, vec), outputVectors)



print "==== Gradient check for skip-gram ===="
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

print "\n==== Gradient check for CBOW      ===="
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)



