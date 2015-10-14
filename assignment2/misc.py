##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    lowerBound = -sqrt(6) / sqrt(m+n)
    upperBound = -lowerBound

    A0 = random.uniform(lowerBound, upperBound, (m,n))
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0