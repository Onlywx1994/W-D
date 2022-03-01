import numpy as np
from input_layer import DenseInputCombineLayer
from embedding_layer import EmbeddingCombineLayer
from dense_layer import DenseLayer
from activation import ReLU
import utils
import logging
from collections import namedtuple
from base_estimator import BaseEstimator

class DeepNetWork:
    def __init__(self,dense_fields,vocab_infos,embed_fields,hidden_units,L2,optimizer):

        self._optimizer=optimizer
        self._dense_combine_layer=DenseInputCombineLayer(dense_fields)
        self._embed_combine_layer=EmbeddingCombineLayer(vocab_infos)
        for field_name,vocab_name in embed_fields:
            self._embed_combine_layer.add_embedding(vocab_name=vocab_name,field_name=field_name)

        self._optimizer_layers=[self._embed_combine_layer]

        prev_out_dim=self._dense_combine_layer.output_dim+self._embed_combine_layer.output_dim
        self._hidden_layers=[]

        for layer_idx,n_units in enumerate(hidden_units,start=1):
            hidden_layer=DenseLayer(name="hidden{}".format(layer_idx),shape=[prev_out_dim,n_units],l2reg=L2)
            self._hidden_layers.append(hidden_layer)
            self._optimizer_layers.append(hidden_layer)

            logging.info("{}-th hidden layer,weight shape ={}".format(layer_idx,hidden_layer.shape))
            self._hidden_layers.append(ReLU())

            prev_out_dim=n_units

        final_logit_layer=DenseLayer(name="final_logit",shape=[prev_out_dim,1],l2reg=L2)

        logging.info("final logit layer ,weight shape={}".format(final_logit_layer.shape))
        self._hidden_layers.append(final_logit_layer)
        self._optimizer_layers.append(final_logit_layer)
        print(self._optimizer_layers[1].variables)

    def forward(self,features):
        dense_input=self._dense_combine_layer.forward(features)
        embed_input=self._embed_combine_layer.forward(features)
        X=np.hstack([dense_input,embed_input])

        for hidden_layer in self._hidden_layers:
            X=hidden_layer.forward(X)
        return X.flatten()

    def backward(self,grads2logits):
        prev_grads=grads2logits.reshape([-1,1])
        for hidden_layer in self._hidden_layers[::-1]:
            prev_grads=hidden_layer.backward(prev_grads)

        col_sizes=[self._dense_combine_layer.output_dim,self._embed_combine_layer.output_dim]

        _,grads_for_all_embedding =utils.split_column(prev_grads,col_sizes)

        self._embed_combine_layer.backward(grads_for_all_embedding)

        all_vars,all_grads2var={},{}
        for opt_layer in self._optimizer_layers:
            all_vars.update(opt_layer.variables)
            all_grads2var.update(opt_layer.grads2var)

        self._optimizer.update(variables=all_vars,gradients=all_grads2var)

DeepHparams=namedtuple("DeepHParams",
                       ["dense_fields","vocab_infos","embed_fields","hidden_units","L2","optimizer"])



class DeepEstimator(BaseEstimator):
    def __init__(self,hparams,data_source):
        self._dnn=DeepNetWork(dense_fields=hparams.dense_fields,
                              vocab_infos=hparams.vocab_infos,
                              embed_fields=hparams.embed_fields,
                              hidden_units=hparams.hidden_units,
                              L2=hparams.L2,
                              optimizer=hparams.optimizer)
        super().__init__(data_source)

    def train_batch(self,features,labels):
        logits=self._dnn.forward(features)
        pred_probs=1/(1+np.exp(-logits))

        grads2logits=pred_probs-labels
        self._dnn.backward(grads2logits)
        return pred_probs

    def predict(self,features):
        logits=self._dnn.forward(features)
        return 1/(1+np.exp(-logits))


