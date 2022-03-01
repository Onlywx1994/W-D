import numpy as np
import initialization

class DenseLayer:
    def __init__(self,name,shape,l2reg=0,init_method='glorot_uniform'):
        self._name=name
        self._l2reg=l2reg
        self._W=initialization.get_global_init(init_method)(shape)
        self._b=initialization.get_global_init('zero')(shape[1])
        self._dW=None
        self._db=None

        self._last_input=None

    def forward(self,X):
        self._last_input=X
        return np.dot(self._last_input,self._W)+self._b

    def backward(self,prev_grads):
        assert prev_grads.shape[1]==self._W.shape[1]

        self._dW=np.dot(self._last_input.T,prev_grads)
        self._dW+=self._l2reg*self._W

        self._db=np.sum(prev_grads,axis=0)
        return  np.dot(prev_grads,self._W.T)

    @property
    def l2reg_loss(self):
        return -.5*self._l2reg*np.sum(self._W**2)

    @property
    def shape(self):
        return self._W.shape

    @property
    def output_dim(self):
        return self._W.shape[1]
    @property
    def variables(self):
        return {"{}_W".format(self._name):self._W,
                "{}_b".format(self._name):self._b}

    @property
    def grads2var(self):
        return {"{}_W".format(self._name):self._dW,
                "{}_b".format(self._name):self._db}