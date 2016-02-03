from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot, sigmoid
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
        self.alpha = alpha # default training rate

        #wv.shape: (100232,50)
        dims[0] = windowsize * wv.shape[1] # input dimension 3*50=150
        param_dims = dict(W=(dims[1], dims[0]),# W(100,150)
                          b1=(dims[1],),#b(100)
                          U=(dims[2], dims[1]),#U(5,100)
                          b2=(dims[2],),#(5,)
                          )
        param_dims_sparse = dict(L=wv.shape) #L(100232,50)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        #### YOUR CODE HERE ####

        # any other initialization you need
        self.sparams.L = wv.copy() # store own representations,100232,50 matrix
        self.params.W = random_weight_matrix(*self.params.W.shape)
        self.params.U = random_weight_matrix(*self.params.U.shape)

        self.window_size = windowsize#3
        self.word_vec_size = wv.shape[1]#50

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
        x = hstack([self.sparams.L[idx] for idx in window]) # extract representation,(150,) matrix
        a =self.params.W.dot(x)+self.params.b1#(100,150)*(150,)+(100,)=>(100,)
        h = tanh(a)#(100,)
        p = softmax(self.params.U.dot(h) + self.params.b2)#(5,100)*(100,)+(100,)=>(5,)
        
        # Compute gradients w.r.t cross-entropy loss
        y = make_onehot(label, len(p))
        delta = p - y #(5,)
        ##
        # Backpropagation
        # dJ/dh
        dh = self.params.U.T.dot(delta) #(100,5)*(5,)=>(100,)
        # dJ/da
        da = dh * (1-tanh(a)**2)#(100,) right
        L_updatevalue=self.params.W.T.dot(da)#(150,100)*(100,)=>(150,)
        
        # dJ/dU, dJ/db2
        self.grads.U += outer(delta, h) + self.lreg * self.params.U#(5,100)
        self.grads.b2 += delta#(5,)

        # dJ/dW, dJ/db1
        self.grads.W += outer(da,x) + self.lreg * self.params.W #(100,)*(150,)+(100,150)=>(100,150)
        self.grads.b1 += da #(100,)
        
        # dJ/dL, sparse update: use sgrads
        dL = self.params.W.T.dot(da).reshape(self.window_size, self.word_vec_size)#(150,100)*(100,)=>(150,)=>(3,50)
        for i in xrange(self.window_size):
            self.sgrads.L[window[i], :] = dL[i]
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
        #print 'windows.shape',windows[0]
        P=[]
        for window in windows:
            x = hstack([self.sparams.L[idx] for idx in window]) # extract representation,(150,) matrix
            #x=reshape(x,(x.shape[0]*x.shape[1]))
            #print self.params.W.shape,' ',x.shape,' ',self.params.b1.shape
            a =self.params.W.dot(x)+self.params.b1#(100,150)*(150,)+(100,)=>(100,)
            h = tanh(a)#(100,)
            p = softmax(self.params.U.dot(h) + self.params.b2)#(5,100)*(100,)+(100,)=>(5,)
            P.append(p)
        #### END YOUR CODE ####


        return P # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####
        c = argmax(self.predict_proba(windows), axis=1)
        #### END YOUR CODE ####
        return c # list of predicted classes


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """
        
        labels_list = None
        #### YOUR CODE HERE ####
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]
            labels_list = [labels]
        else:
            labels_list = labels

        J = 0
        for i in xrange(len(windows)):
            x = hstack(self.sparams.L[windows[i], :]) # extract representation
            h = tanh(self.params.W.dot(x) + self.params.b1)
            p = softmax(self.params.U.dot(h) + self.params.b2)
            J += - log(p[labels_list[i]])
        Jreg = (self.lreg / 2.0) * (sum(self.params.W**2.0) + sum(self.params.U**2.0))
        #### END YOUR CODE ####
        
        return J + Jreg