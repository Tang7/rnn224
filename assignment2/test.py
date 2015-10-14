from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot, sigmoid
from misc import random_weight_matrix

import data_utils.utils as du
import data_utils.ner as ner
from misc import random_weight_matrix

import itertools

alphaiter = itertools.repeat(0.1)
a = itertools.izip(alphaiter)
print a