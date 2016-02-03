
# coding: utf-8

# # CS 224D Assignment #2
# # Part [1]: Deep Networks: NER Window Model
# 
# For this first part of the assignment, you'll build your first "deep" networks. On problem set 1, you computed the backpropagation gradient $\frac{\partial J}{\partial w}$ for a two-layer network; in this problem set you'll implement a slightly more complex network to perform  named entity recognition (NER).
# 
# Before beginning the programming section, you should complete parts (a) and (b) of the corresponding section of the handout.

# In[1]:

import sys, os
from numpy import *
from matplotlib.pyplot import *
get_ipython().magic(u'matplotlib inline')
matplotlib.rcParams['savefig.dpi'] = 100

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# ## (c): Random Initialization Test
# Use the cell below to test your code.

# In[2]:

from misc import random_weight_matrix
random.seed(10)
print random_weight_matrix(3,5)


# ## (d): Implementation
# 
# We've provided starter code to load in the dataset and convert it to a list of "windows", consisting of indices into the matrix of word vectors. 
# 
# We pad each sentence with begin and end tokens `<s>` and `</s>`, which have their own word vector representations; additionally, we convert all words to lowercase, canonicalize digits (e.g. `1.12` becomes `DG.DGDG`), and replace unknown words with a special token `UUUNKKK`.
# 
# You don't need to worry about the details of this, but you can inspect the `docs` variables or look at the raw data (in plaintext) in the `./data/` directory.

# In[3]:

import data_utils.utils as du
import data_utils.ner as ner


# In[4]:

# Load the starter word vectors
wv, word_to_num, num_to_word = ner.load_wv('data/ner/vocab.txt',
                                           'data/ner/wordVectors.txt')
#wv:(100232,50)
#word_to_num:dict,100232
tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
num_to_tag = dict(enumerate(tagnames))
tag_to_num = du.invert_dict(num_to_tag)

# Set window size
windowsize = 3

# Load the training set
docs = du.load_dataset('data/ner/train')
X_train, y_train = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                      wsize=windowsize)

# Load the dev set (for tuning hyperparameters)
docs = du.load_dataset('data/ner/dev')
X_dev, y_dev = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                  wsize=windowsize)

# Load the test set (dummy labels only)
docs = du.load_dataset('data/ner/test.masked')
X_test, y_test = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                    wsize=windowsize)


# In[5]:

print num_to_tag
print tag_to_num
print wv.shape
print len(word_to_num)
print type(word_to_num)
print docs[0][:]
print X_train[0]
print y_train[0]


# To avoid re-inventing the wheel, we provide a base class that handles a lot of the drudgery of managing parameters and running gradient descent. It's based on the classifier API used by [`scikit-learn`](http://scikit-learn.org/stable/), so if you're familiar with that library it should be easy to use. 
# 
# We'll be using this class for the rest of this assignment, so it helps to get acquainted with a simple example that should be familiar from Assignment 1. To keep this notebook uncluttered, we've put the code in the `softmax_example.py`; take a look at it there, then run the cell below.

# In[71]:

from softmax_example import SoftmaxRegression
sr = SoftmaxRegression(wv=zeros((10,100)), dims=(100,5))

##
# Automatic gradient checker!
# this checks anything you add to self.grads or self.sgrads
# using the method of Assignment 1

sr.grad_check(x=5, y=0)

# sr._acc_grads(idx=5,label=4)
# print sr.params.W.shape
# print sr.params.b.shape
# print sr.sparams.L.shape
# # print sr.sgrads.L[1].shape

# sr.compute_loss(idx=5,label=4)


# In order to implement a model, you need to subclass `NNBase`, then implement the following methods:
# 
# - `__init__()` (initialize parameters and hyperparameters)
# - `_acc_grads()` (compute and accumulate gradients)
# - `compute_loss()` (compute loss for a training example)
# - `predict()`, `predict_proba()`, or other prediction method (for evaluation)
# 
# `NNBase` provides you with a few others that will be helpful:
# 
# - `grad_check()` (run a gradient check - calls `_acc_grads` and `compute_loss`)
# - `train_sgd()` (run SGD training; more on this later)
# 
# Your task is to implement the window model in `nerwindow.py`; a scaffold has been provided for you with instructions on what to fill in.
# 
# When ready, you can test below:

# In[72]:

wv.shape


# In[5]:

from nerwindow import WindowMLP
clf = WindowMLP(wv, windowsize=windowsize, dims=[None, 100, 5],
                reg=0.001, alpha=0.01)
# clf._acc_grads(X_train[0],y_train[0])
clf.grad_check(X_train[0], y_train[0]) # gradient check on single point


# Now we'll train your model on some data! You can implement your own SGD method, but we recommend that you just call `clf.train_sgd`. This takes the following arguments:
# 
# - `X`, `y` : training data
# - `idxiter`: iterable (list or generator) that gives index (row of X) of training examples in the order they should be visited by SGD
# - `printevery`: int, prints progress after this many examples
# - `costevery`: int, computes mean loss after this many examples. This is a costly operation, so don't make this too frequent!
# 
# The implementation we give you supports minibatch learning; if `idxiter` is a list-of-lists (or yields lists), then gradients will be computed for all indices in a minibatch before modifying the parameters (this is why we have you write `_acc_grad` instead of applying them directly!).
# 
# Before training, you should generate a training schedule to pass as `idxiter`. If you know how to use Python generators, we recommend those; otherwise, just make a static list. Make the following in the cell below:
# 
# - An "epoch" schedule that just iterates through the training set, in order, `nepoch` times.
# - A random schedule of `N` examples sampled with replacement from the training set.
# - A random schedule of `N/k` minibatches of size `k`, sampled with replacement from the training set.

# In[121]:

print X_train.shape
print type(xrange(5))
print 4*range(4)


# In[7]:

nepoch = 5
N = nepoch * len(y_train)#5*203621
k = 5 # minibatch size

random.seed(10) # do not change this!
#### YOUR CODE HERE ####
indices = range(len(y_train))

#plan1:An "epoch" schedule that just iterates through the training set, in order, nepoch times
idxiter_epoch = nepoch * indices#solid sequence 12341234

#plan2:A random schedule of N examples sampled with replacement from the training set.
idxiter_N = random.choice(indices, N)

#plan3:A random schedule of N/k minibatches of size k, sampled with replacement from the training set.
def idxiter_batches():
    num_batches = N / k
    for i in xrange(num_batches):
        yield random.choice(indices, k)#random.choice(range(2w),5)

#### END YOUR CODE ###


# Now call `train_sgd` to train on `X_train`, `y_train`. To verify that things work, train on 100,000 examples or so to start (with any of the above schedules). This shouldn't take more than a couple minutes, and you should get a mean cross-entropy loss around 0.4.
# 
# Now, if this works well, it's time for production! You have three tasks here:
# 
# 1. Train a good model
# 2. Plot a learning curve (cost vs. # of iterations)
# 3. Use your best model to predict the test set
# 
# You should train on the `train` data and evaluate performance on the `dev` set. The `test` data we provided has only dummy labels (everything is `O`); we'll compare your predictions to the true labels at grading time. 
# 
# Scroll down to section (f) for the evaluation code.
# 
# We don't expect you to spend too much time doing an exhaustive search here; the default parameters should work well, although you can certainly do better. Try to achieve an F1 score of at least 76% on the dev set, as reported by `eval_performance`.
# 
# Feel free to create new cells and write new code here, including new functions (helpers and otherwise) in `nerwindow.py`. When you have a good model, follow the instructions below to make predictions on the test set.
# 
# A strong model may require 10-20 passes (or equivalent number of random samples) through the training set and could take 20 minutes or more to train - but it's also possible to be much, much faster!
# 
# Things you may want to tune:
# - `alpha` (including using an "annealing" schedule to decrease the learning rate over time)
# - training schedule and minibatch size
# - regularization strength
# - hidden layer dimension
# - width of context window

# In[120]:

#### YOUR CODE HERE ####
# Sandbox: build a good model by tuning hyperparameters
clf.train_sgd(X_train[:100000], y_train[:100000], idxiter=xrange(100000), printevery=5000, costevery=5000,)

#### END YOUR CODE ####


# In[ ]:

#### YOUR CODE HERE ####
# Sandbox: build a good model by tuning hyperparameters
from nerwindow import full_report, eval_performance

schedules = [idxiter_epoch, idxiter_N, idxiter_batches()]#last one is best, but slow. choose second
for train_idxiter in schedules:
    clf = WindowMLP(wv, windowsize=windowsize, dims=[None, 100, 5], reg=0.001, alpha=0.01)
    clf.train_sgd(X_train, y_train, idxiter=train_idxiter, printevery=250000, costevery=250000,)
    yp = clf.predict(X_dev)
    full_report(y_dev, yp, tagnames) # full report, helpful diagnostics
    eval_performance(y_dev, yp, tagnames) # performance: optimize this F1

#### END YOUR CODE ####


# In[8]:

#### YOUR CODE HERE ####
# Sandbox: build a good model by tuning hyperparameters
from nerwindow import full_report, eval_performance

regs = [0.0005, 0.0001, 0.00005]#0.0001 is best
for reg in regs:
    print "===== Regularization %f:" % reg
    clf = WindowMLP(wv, windowsize=windowsize, dims=[None, 100, 5], reg=reg, alpha=0.01)
    clf.train_sgd(X_train, y_train, idxiter=idxiter_batches(), printevery=10000, costevery=10000,)
    yp = clf.predict(X_dev)
    full_report(y_dev, yp, tagnames) # full report, helpful diagnostics
    eval_performance(y_dev, yp, tagnames) # performance: optimize this F1

#### END YOUR CODE ####


# In[12]:

wv.shape


# In[16]:

from nerwindow import full_report, eval_performance
window_sizes = [5, 7, 9]#7 is best
for window_size in window_sizes:
    
    # Load the training set
    X_train, y_train = du.docs_to_windows(du.load_dataset('data/ner/train'), word_to_num, tag_to_num, wsize=window_size)
    # Load the dev set (for tuning hyperparameters)
    X_dev, y_dev = du.docs_to_windows(du.load_dataset('data/ner/dev'), word_to_num, tag_to_num, wsize=window_size)
    
    print "===== window_size %f:" % window_size
    clf = WindowMLP(wv, windowsize=window_size, dims=[None, 100, 5], reg=0.0001, alpha=0.01)
    clf.train_sgd(X_train, y_train, idxiter=idxiter_N, printevery=250000, costevery=250000,)
    yp = clf.predict(X_dev)
    full_report(y_dev, yp, tagnames) # full report, helpful diagnostics
    eval_performance(y_dev, yp, tagnames) # performance: optimize this F1


# In[19]:

#using inverse decay to change learning rate
gamma=0.0001
power=0.75
base_lr=0.01
def alphaiter_inv(gamma=0.0001,power=0.75,base_lr=0.01):
    i=0
    while i<N:
        i+=1
        lr=base_lr*(1.0+gamma*i)**(-power)
        yield lr
        


# In[25]:

import matplotlib.pyplot as plt
lr_values_inv=np.array(list(alphaiter_inv()),dtype='float_')
lr_values_inv.shape
plt.plot(range(len(lr_values_inv)),lr_values_inv)


# In[26]:

reg=0.0001
window_size=7

# Load the training set
X_train, y_train = du.docs_to_windows(du.load_dataset('data/ner/train'), word_to_num, tag_to_num, wsize=window_size)
# Load the dev set (for tuning hyperparameters)
X_dev, y_dev = du.docs_to_windows(du.load_dataset('data/ner/dev'), word_to_num, tag_to_num, wsize=window_size)

clf = WindowMLP(wv, windowsize=window_size, dims=[None, 100, 5], reg=0.0001)
clf.train_sgd(X_train, y_train, idxiter=idxiter_N, alphaiter=alphaiter_inv(), printevery=250000, costevery=250000,)
yp = clf.predict(X_dev)
full_report(y_dev, yp, tagnames) # full report, helpful diagnostics
eval_performance(y_dev, yp, tagnames) # performance: optimize this F1


# In[9]:

from nerwindow import full_report, eval_performance
reg=0.0001
window_size=7

# Load the training set
X_train, y_train = du.docs_to_windows(du.load_dataset('data/ner/train'), word_to_num, tag_to_num, wsize=window_size)
# Load the dev set (for tuning hyperparameters)
X_dev, y_dev = du.docs_to_windows(du.load_dataset('data/ner/dev'), word_to_num, tag_to_num, wsize=window_size)

clf = WindowMLP(wv, windowsize=window_size, dims=[None, 100, 5], reg=0.0001,alpha=0.01)
traincurvebest = clf.train_sgd(X_train, y_train, idxiter=idxiter_N, printevery=250000, costevery=250000,)
yp = clf.predict(X_dev)
full_report(y_dev, yp, tagnames) # full report, helpful diagnostics
eval_performance(y_dev, yp, tagnames) # performance: optimize this F1


# ## (e): Plot Learning Curves
# The `train_sgd` function returns a list of points `(counter, cost)` giving the mean loss after that number of SGD iterations.
# 
# If the model is taking too long you can cut it off by going to *Kernel->Interrupt* in the IPython menu; `train_sgd` will return the training curve so-far, and you can restart without losing your training progress.
# 
# Make two plots:
# 
# - Learning curve using `reg = 0.001`, and comparing the effect of changing the learning rate: run with `alpha = 0.01` and `alpha = 0.1`. Use minibatches of size 5, and train for 10,000 minibatches with `costevery=200`. Be sure to scale up your counts (x-axis) to reflect the batch size. What happens if the model tries to learn too fast? Explain why this occurs, based on the relation of SGD to the true objective.
# 
# - Learning curve for your best model (print the hyperparameters in the title), as trained using your best schedule. Set `costevery` so that you get at least 100 points to plot.

# In[47]:

traincurvebest


# In[29]:

##
# Plot your best learning curve here
counts, costs = zip(*traincurvebest)
figure(figsize=(6,4))
plot(5*array(counts), costs, color='b', marker='o', linestyle='-')
title(r"Learning Curve ($\alpha$=%g, $\lambda$=%g)" % (clf.alpha, clf.lreg))
xlabel("SGD Iterations"); ylabel(r"Average $J(\theta)$"); 
ylim(ymin=0, ymax=max(1.1*max(costs),3*min(costs)));
ylim(0,0.5)

# Don't change this filename!
savefig("ner.learningcurve.best.png")


# In[30]:

##
# Plot comparison of learning rates here
# feel free to change the code below

window_size = 3
# Load the training set
docs = du.load_dataset('data/ner/train')
X_train, y_train = du.docs_to_windows(docs, word_to_num, tag_to_num, wsize=window_size)

# Load the dev set (for tuning hyperparameters)
docs = du.load_dataset('data/ner/dev')
X_dev, y_dev = du.docs_to_windows(docs, word_to_num, tag_to_num, wsize=window_size)

def idxiter_batches1():
    num_batches = 10000
    for i in xrange(num_batches):
        yield random.choice(indices, 5)

print X_train.shape, y_train.shape
print X_dev.shape, y_dev.shape

clf1 = WindowMLP(wv, windowsize=window_size, dims=[None, 100, 5], reg=0.001, alpha=0.01)
trainingcurve1 = clf1.train_sgd(X_train, y_train, idxiter=idxiter_batches1(), printevery=50000, costevery=200)
clf2 = WindowMLP(wv, windowsize=window_size, dims=[None, 100, 5], reg=0.001, alpha=0.1)
trainingcurve2 = clf2.train_sgd(X_train, y_train, idxiter=idxiter_batches1(), printevery=50000, costevery=200)


figure(figsize=(6,4))
counts, costs = zip(*trainingcurve1)
plot(5*array(counts), costs, color='b', marker='o', linestyle='-', label=r"$\alpha=0.01$")
counts, costs = zip(*trainingcurve2)
plot(5*array(counts), costs, color='g', marker='o', linestyle='-', label=r"$\alpha=0.1$")
title(r"Learning Curve ($\lambda=0.01$, minibatch k=5)")
xlabel("SGD Iterations"); ylabel(r"Average $J(\theta)$"); 
ylim(ymin=0, ymax=max(1.1*max(costs),3*min(costs)));
legend()

# Don't change this filename
savefig("ner.learningcurve.comparison.png")


# ## (f): Evaluating your model
# Evaluate the model on the dev set using your `predict` function, and compute performance metrics below!

# In[49]:

window_size=7
# Load the training set
X_train, y_train = du.docs_to_windows(du.load_dataset('data/ner/train'), word_to_num, tag_to_num, wsize=window_size)
# Load the dev set (for tuning hyperparameters)
X_dev, y_dev = du.docs_to_windows(du.load_dataset('data/ner/dev'), word_to_num, tag_to_num, wsize=window_size)
# Predict labels on the dev set
yp = clf.predict(X_dev)
# Save predictions to a file, one per line
ner.save_predictions(yp, "dev.predicted")


# In[51]:

from nerwindow import full_report, eval_performance
full_report(y_dev, yp, tagnames) # full report, helpful diagnostics
eval_performance(y_dev, yp, tagnames) # performance: optimize this F1


# In[57]:

# Save your predictions on the test set for us to evaluate
# IMPORTANT: make sure X_test is exactly as loaded 
# from du.docs_to_windows, so that your predictions 
# line up with ours.
# Load the test set (dummy labels only)
window_size=7
docs = du.load_dataset('data/ner/test.masked')
X_test, y_test = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                    wsize=window_size)

yptest = clf.predict(X_test)
ner.save_predictions(yptest, "test.predicted")


# ## Part [1.1]: Probing neuron responses
# 
# You might have seen some results from computer vision where the individual neurons learn to detect edges, shapes, or even [cat faces](http://googleblog.blogspot.com/2012/06/using-large-scale-brain-simulations-for.html). We're going to do the same for language.
# 
# Recall that each "neuron" is essentially a logistic regression unit, with weights corresponding to rows of the corresponding matrix. So, if we have a hidden layer of dimension 100, then we can think of our matrix $W \in \mathbb{R}^{100 x 150}$ as representing 100 hidden neurons each with weights `W[i,:]` and bias `b1[i]`.
# 
# ### (a): Hidden Layer, Center Word
# For now, let's just look at the center word, and ignore the rest of the window. This corresponds to columns `W[:,50:100]`, although this could change if you altered the window size for your model. For each neuron, find the top 10 words that it responds to, as measured by the dot product between `W[i,50:100]` and `L[j]`. Use the provided code to print these words and their scores for 5 neurons of your choice. In your writeup, briefly describe what you notice here.
# 
# The `num_to_word` dictionary, loaded earlier, may be helpful.

# In[11]:

# Recommended function to print scores
# scores = list of float
# words = list of str
def print_scores(scores, words):
    for i in range(len(scores)):
        print "[%d]: (%.03f) %s" % (i, scores[i], words[i])


# In[72]:

# Recommended function to print scores
# scores = list of float
# words = list of str
def print_scores(scores, words):
    for i in range(len(scores)):
        print "[%d]: (%.03f) %s" % (i, scores[i], words[i])

#### YOUR CODE HERE ####

neurons = [1,3,4,6,8] # change this to your chosen neurons
for i in neurons:
    print "Neuron %d" % i
    
    # W[...] is 1 x 50, L is |V| x 50
    # get the argmax indices of this:
    dots = np.dot(clf.params.W[i, 150:200], clf.sparams.L.T)#(50,)(50,20w)=>(20w,)
    indices = dots.argsort()[-10:][::-1]
    words = [num_to_word[i] for i in indices]
    scores = [dots[i] for i in indices]
    
    print_scores(scores,words)
    
#### END YOUR CODE ####


# ### (b): Model Output, Center Word
# Now, let's do the same for the output layer. Here we only have 5 neurons, one for each class. `O` isn't very interesting, but let's look at the other four.
# 
# Here things get a little more complicated: since we take a softmax, we can't just look at the neurons separately. An input could cause several of these neurons to all have a strong response, so we really need to compute the softmax output and find the strongest inputs for each class.
# 
# As before, let's consider only the center word (`W[:,50:100]`). For each class `ORG`, `PER`, `LOC`, and `MISC`, find the input words that give the highest probability $P(\text{class}\ |\ \text{word})$.
# 
# You'll need to do the full feed-forward computation here - for efficiency, try to express this as a matrix operation on $L$. This is the same feed-forward computation as used to predict probabilities, just with $W$ replaced by `W[:,50:100]`.
# 
# As with the hidden-layer neurons, print the top 10 words and their corresponding class probabilities for each class.

# In[95]:

#### YOUR CODE HERE ####
from nn.math import softmax
topscores=[]
topwords=[]
for i in range(1,5):
    print "Output neuron %d: %s" % (i, num_to_tag[i])
    dots = np.dot(clf.params.W[:, 150:200], clf.sparams.L.T)+clf.params.b1.reshape((100,1))#(100,50)(50,20w)=>(100,20w)
    h = tanh(dots)#(100,)
    a2=clf.params.U.dot(h) + clf.params.b2.reshape((5,1))#(5,100)*(100,20w)+(5,)=>(5,20w)
    a2=a2.T
    
    N,C=a2.shape
    x_exp = np.exp(a2 - np.max(a2, axis=1).reshape(N, 1))
    p = x_exp / np.sum(x_exp, axis=1).reshape(N, 1)
    indices = p[:, i].argsort()[-10:][::-1]
    print 'indices:',indices
    words=[num_to_word[j] for j in indices]
    scores=[p[j,i] for j in indices]
    topscores.append(scores)
    topwords.append(words)

for i in range(1,5):
    print "Output neuron %d: %s" % (i, num_to_tag[i])
    print_scores(topscores[i-1], topwords[i-1])
    print ""

#### END YOUR CODE ####


# ### (c): Model Output, Preceding Word
# Now for one final task: let's look at the preceding word. Repeat the above analysis for the output layer, but use the first part of $W$, i.e. `W[:,:50]`.
# 
# Describe what you see, and include these results in your writeup.

# In[12]:

#### YOUR CODE HERE ####
h = tanh(clf.sparams.L.dot(clf.params.W[:, :50].T) + clf.params.b1)
a = h.dot(clf.params.U.T) + clf.params.b2
N, C = a.shape
x_exp = np.exp(a - np.max(a, axis=1).reshape(N, 1))
p = x_exp / np.sum(x_exp, axis=1).reshape(N, 1)
for i in range(1,5):
    print "Output neuron %d: %s" % (i, num_to_tag[i])
    indices = p[:, i].argsort()[-10:][::-1]
    print indices
    words = [num_to_word[j] for j in indices]
    scores = [p[j][i] for j in indices]
    print_scores(scores, words)
    print ""

#### END YOUR CODE ####


# In[ ]:



