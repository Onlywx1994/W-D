import numpy as np
from initialization import TruncatedNormal
import utils


class EmbeddingLayer:

    def __init__(self, W, vocab_name, field_name):
        self.vocab_name = vocab_name
        self.field_name = field_name
        self._W = W
        self._last_input = None

    @property
    def output_dim(self):
        return self._W.shape[1]

    def forward(self, X):
        self._last_input = X
        output = np.zeros((X.n_total_examples, self._W.shape[1]))

        for example_idx, feat_id, feat_val in X.iterate_non_zeros():
            embedding = self._W[feat_id, :]
            output[example_idx, :] += embedding * feat_val
        return output

    def backward(self, prev_grads):

        dW = {}

        for example_idx, feat_id, feat_val in self._last_input.iterate_non_zeros():
            grad_from_one_example = prev_grads[example_idx, :] * feat_val

            if feat_id in dW:
                dW[feat_id] += grad_from_one_example
            else:
                dW[feat_id] = grad_from_one_example

        return dW


class EmbeddingCombineLayer:
    def __init__(self, vocab_infos):

        self._weights = {}
        for vocab_name, vocab_size, embed_size in vocab_infos:
            stddev = 1 / np.sqrt(embed_size)
            initializer = TruncatedNormal(mean=0,
                                          stddev=stddev,
                                          lower=-2 * stddev,
                                          upper=2 * stddev)
            self._weights[vocab_name] = initializer(shape=[vocab_size, embed_size])

        self._grad_to_embed = {}
        self._embed_layers = []

    def add_embedding(self, vocab_name, field_name):
        weight = self._weights[vocab_name]
        layer = EmbeddingLayer(W=weight, vocab_name=vocab_name, field_name=field_name)
        self._embed_layers.append(layer)

    @property
    def output_dim(self):
        return sum(layer.output_dim for layer in self._embed_layers)

    def forward(self, sparse_inputs):
        embedded_outputs = []
        for embed_layer in self._embed_layers:
            sp_input = sparse_inputs[embed_layer.field_name]
            embedded_outputs.append(embed_layer.forward(sp_input))
        return np.hstack(embedded_outputs)

    def backward(self,prev_grads):

        assert prev_grads.shape[1]==self.output_dim


        col_size=[layer.output_dim for layer in self._embed_layers]
        prev_grads_splits=utils.split_column(prev_grads,col_size)

        self._grad_to_embed.clear()
        for layer,layer_prev_grads in zip(self._embed_layers,prev_grads_splits):
            layer_grads_to_embed=layer.backward(layer_prev_grads)

            for feat_id,g in layer_grads_to_embed.items():
                key ="{}@{}".format(layer.vocab_name,feat_id)

                if key in self._grad_to_embed:
                    self._grad_to_embed[key]+=g

                else:
                    self._grad_to_embed[key]=g

    @property
    def variables(self):
        return self._weights

    @property
    def grads2var(self):
        return self._grad_to_embed

    @property
    def l2reg_loss(self):
        return 0