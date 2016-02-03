import numpy as np
import collections
import pdb


# This is a 2-Layer Deep Recursive Neural Netowrk with two ReLU Layers and a softmax layer
# You must update the forward and backward propogation functions of this file.

# You can run this file via 'python rnn2deep.py' to perform a gradient check

# tip: insert pdb.set_trace() in places where you are unsure whats going on


class RNN2_drop:

    def __init__(self,wvecDim, middleDim, outputDim,numWords,dropoutFraction=0.5,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.middleDim = middleDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho
        
        #dropout fraction
        self.dropoutFraction=dropoutFraction

    def initParams(self):
        np.random.seed(12341)

        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights for layer 1
        self.W1 = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.b1 = np.zeros((self.wvecDim))

        # Hidden activation weights for layer 2
        self.W2 = 0.01*np.random.randn(self.middleDim,self.wvecDim)
        self.b2 = np.zeros((self.middleDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.middleDim) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs]

        # Gradients
        self.dW1 = np.empty(self.W1.shape)
        self.db1 = np.empty((self.wvecDim))
        
        self.dW2 = np.empty(self.W2.shape)
        self.db2 = np.empty((self.middleDim))

        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))
        
        #dropout mask
        self.dropoutMasks=[] #empty list
       
        


    def costAndGrad(self,mbdata,test=False): 
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W1, W2, Ws, b1, b2, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns 
           cost, correctArray, guessArray, total
        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs = self.stack
        # Zero gradients
        self.dW1[:] = 0
        self.db1[:] = 0
        
        self.dW2[:] = 0
        self.db2[:] = 0

        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        np.random.seed(12341)
        temp=0
        for tree in mbdata:
            #dropoutMask = np.random.rand(node.hActs2.shape[0])<self.dropoutFraction
            dropoutMask = np.random.rand(self.dW2.shape[0])<self.dropoutFraction
            self.dropoutMasks.append(dropoutMask);
            c,tot = self.forwardProp(tree.root,correct,guess,test,dropoutMask)
            cost += c
            total += tot
            temp+=1
            
        if test:
            return (1./len(mbdata))*cost,correct, guess, total

        # Back prop each tree in minibatch
        temp=0
        for tree in mbdata:
            self.backProp(tree.root,index_dropoutmask=temp)
            temp += 1

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.W1**2)
        cost += (self.rho/2)*np.sum(self.W2**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dW1 + self.rho*self.W1),scale*self.db1,
                                   scale*(self.dW2 + self.rho*self.W2),scale*self.db2,
                                   scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]

    def forwardProp(self,node, correct=[], guess=[],test=False,dropoutMask=[]):
        cost  =  total = 0.0
        # this is exactly the same setup as forwardProp in rnn.py
        
        # if we are in a leaf node, set hActs1 to be the word vector
        if node.isLeaf:
            node.hActs1 = self.L[:, node.word]
        # if haven't finished doing forward prop on the left child, do it
        else:
            if not node.left.fprop:
                cost_left, total_left = self.forwardProp(node.left, correct, guess,test,dropoutMask)
                cost += cost_left
                total += total_left
            if not node.right.fprop:
                cost_right, total_right = self.forwardProp(node.right, correct, guess,test,dropoutMask)
                cost += cost_right
                total += total_right
            node.hActs1 = np.dot(self.W1, np.hstack([node.left.hActs1, node.right.hActs1])) + self.b1#(d,2d)(2d,)+(d,)
            node.hActs1[node.hActs1 < 0] = 0#(d,)
            
        node.hActs2 = np.dot(self.W2, node.hActs1)+self.b2 #(dm,d)(d)+(dm,)=>(dm,)
        node.hActs2[node.hActs2 < 0] = 0#(dm,)
        
        #dropout layer
        if self.dropoutFraction is not None and self.dropoutFraction > 0:
            #pdb.set_trace()
            if test:
                node.hActs2 *= (1-self.dropoutFraction)
            else:
                #drop layer function
                #nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                #nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
                
                node.hActs2[dropoutMask] = 0

        
        node.probs = np.dot(self.Ws, node.hActs2) + self.bs#(5,d)(d,)+(5,)=>(5,)
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs) / np.sum(np.exp(node.probs))

        y_hat = np.argmax(node.probs)
        guess.append(y_hat)

        y = node.label
        correct.append(y)
        cost += - np.log(node.probs[y])

        node.fprop = True
            

        return cost, total + 1

    def backProp(self,node,index_dropoutmask=0,error=None):

        # Clear nodes
        node.fprop = False
        
        

        # this is exactly the same setup as backProp in rnn.py
        # softmax grad
        dsoftmax = node.probs#(5,)
        dsoftmax[node.label] -= 1.0
        
        self.dWs += np.outer(dsoftmax, node.hActs2)#(5,)outer(d,)=>(5,d)
        self.dbs += dsoftmax#(5,)
        
       
        
        #difference between rnn.py
        dh2 = np.dot(self.Ws.T, dsoftmax)#(dm,5)(5,)=>(dm,)
        
        #dropout layer
        #pdb.set_trace()
        dropoutMask=self.dropoutMasks[index_dropoutmask]
        
        #pdb.set_trace()
        dh2[dropoutMask]=0#(5,)
        
        dh2 *= (node.hActs2 != 0)
        self.dW2 += np.outer(dh2,node.hActs1)#(dm,d)
        self.db2 += dh2 #(dm,)
        dh = np.dot(dh2,self.W2)#(dm,)(dm,d)=>(d)
        
        if error is not None:
            dh += error

        dh *= (node.hActs1 != 0)

        if node.isLeaf:
            self.dL[node.word] += dh
        else:
            self.dW1 += np.outer(dh, np.hstack([node.left.hActs1, node.right.hActs1]))#(d,)outer(2d,)=>(d,2d)
            self.db1 += dh

            dh = np.dot(self.W1.T, dh)#upper loss grad on lower h
            #pdb.set_trace()
            self.backProp(node.left,index_dropoutmask,dh[:self.wvecDim])#left child backprop with error from parent
            self.backProp(node.right,index_dropoutmask,dh[self.wvecDim:])#right child backprop with error from parent

        

        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)

        err1 = 0.0
        count = 0.0
        print "Checking dWs, dW1 and dW2..."
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None] # add dimension since bias is flat
            dW = dW[...,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    W[i,j] += epsilon
                    costP,_ = self.costAndGrad(data)
                    W[i,j] -= epsilon
                    numGrad = (costP - cost)/epsilon
                    err = np.abs(dW[i,j] - numGrad)
                    err1+=err
                    count+=1
        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1/count)
        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                err2+=err
                count+=1

        if 0.001 > err2/count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)

if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    middleDim = 10
    outputDim = 5

    rnn = RNN2(wvecDim,middleDim,outputDim,numW,mbSize=4)
    rnn.initParams()

    mbData = train[:4]
    
    print "Numerical gradient check..."
    rnn.check_grad(mbData)






