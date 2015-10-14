from numpy import *
from nn.base import NNBase
from nn.math import sigmoid,softmax,make_onehot
from misc import random_weight_matrix


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))


##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha  # default training rate

        dims[0] = windowsize * wv.shape[1] # input dimension
        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        #### YOUR CODE HERE ####
        # copy from softmax_example.py
        # self.params.<name> for normal parameters
        # self.sparams.<name> for params with sparse gradients
        # and get access to normal NumPy arrays
        self.d = wv.shape[1]        # the dimension of each word vector
        self.sparams.L = wv.copy()  # store own representations
        self.params.W = random_weight_matrix(*self.params.W.shape)
        self.params.U = random_weight_matrix(*self.params.U.shape)
        # self.params.b1 = zeros((self.nclass,1)) # done automatically!
        # self.params.b2 = zeros((self.nclass,1)) # done automatically!
        #### END YOUR CODE ####


    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        #### YOUR CODE HERE ####

        ##
        # Forward propagation
        x = self.sparams.L[window].reshape(-1)
        z1 = self.params.W.dot(x) + self.params.b1  # b1 shape is (dims[1],)
        h = 2.0*sigmoid(2.0*z1) - 1.0
        z2 = self.params.U.dot(h) + self.params.b2
        p = softmax(z2)

        ##
        # Backpropagation
        # Compute gradients w.r.t cross-entropy loss
        y = make_onehot(label, len(p))
        delta2 = p - y
        delta1 = self.params.U.T.dot(delta2) * (1 - h**2.0)

        self.grads.U += outer(delta2, h) + self.lreg * self.params.U
        self.grads.b2 += delta2

        self.grads.W += outer(delta1, x) + self.lreg * self.params.W
        self.grads.b1 += delta1

        self.sgrads.L[window] = self.params.W.T.dot(delta1)
        # update each word one by one 
        dL = self.params.W.T.dot(delta1)
        for i in xrange(len(window)):
            f = dL[i * self.d : (i + 1) * self.d]
            self.sgrads.L[window[i]] = f.reshape((self.d,))

        #### END YOUR CODE ####

    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        #### YOUR CODE HERE ####
        ### Should be processed row by row, because different windows may have common index.
        ### Although the gradient is correct for one window, may be wrong in minibatch training!
        num = len(windows)
        windowsize = len(windows[0])
        X = self.sparams.L[windows].reshape(windowsize * self.sparams.L.shape[1], num)
        # X = hstack(self.sparams.L[windows])
        z1 = self.params.W.dot(X) + self.params.b1.reshape(self.params.b1.shape[0], 1)
        h = 2.0*sigmoid(2.0*z1) - 1.0
        z2 = self.params.U.dot(h) + self.params.b2.reshape(self.params.b2.shape[0], 1)
        P = softmax(z2)
        #### END YOUR CODE ####

        return P  # rows are output for each input

    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####
        P = self.predict_prob(windows)  # numclass x num examples
        c = argmax(P, axis=0)
        #### END YOUR CODE ####
        return c  # list of predicted classes

    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """

        #### YOUR CODE HERE ####
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        if type(labels) is not list:
            labels = [labels]

        num = len(windows)
        windowsize = len(windows[0])
        X = self.sparams.L[windows].reshape(windowsize * self.sparams.L.shape[1], num)
        # X = hstack(self.sparams.L[windows])
        z1 = self.params.W.dot(X) + self.params.b1.reshape(self.params.b1.shape[0], 1)
        h = 2.0*sigmoid(2.0*z1) - 1.0
        z2 = self.params.U.dot(h) + self.params.b2.reshape(self.params.b2.shape[0], 1)
        P = softmax(z2)
				# the shape of P should be nclass x num examples
        J = -sum(log(P[labels, range(0, len(labels))]))

        Jreg = (self.lreg / 2.0) * (sum(self.params.W**2.0) + sum(self.params.U**2.0))

        #### END YOUR CODE ####
        return J+Jreg