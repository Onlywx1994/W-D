import numpy as np
from ftrl import FtrlEstimator
from base_estimator import BaseEstimator
from collections import namedtuple


class WideLayer:
    def __init__(self, field_names, alpha, beta, L1, L2, prob_fn):

        self._estimators = {field: FtrlEstimator(alpha=alpha,
                                                 beta=beta,
                                                 L1=L1,
                                                 L2=L2) for field in (['bias'] + field_names)}

        self._prob_fn = prob_fn

    def __predict_logit(self, sp_features, example_idx):

        logit = 0
        for field, estimator in self._estimators.items():
            if field == "bias":
                feat_ids = [0]
                feat_vals = [1]
            else:
                sp_input = sp_features[field]
                feat_ids, feat_vals = sp_input.get_example_in_order(example_idx)

            logit += estimator.predict_logit(feature_ids=feat_ids, feature_values=feat_vals)

        return logit

    def train(self, sp_features, labels):

        probs = []
        for example_idx, label in enumerate(labels):
            logit = self.__predict_logit(sp_features, example_idx)
            pred_prob = self._prob_fn(example_idx, logit)
            probs.append(pred_prob)

            for _, estimator in self._estimators.items():
                estimator.update(pred_prob=pred_prob, label=label)
        return np.asarray(probs)

    def predict_logits(self, sp_features):
        batch_size = None
        for sp_input in sp_features.values():
            batch_size = sp_input.n_total_examples
            break
        logits = [self.__predict_logit(sp_features, example_idx) for example_idx in range(batch_size)]
        return np.asarray(logits)


WideHparams = namedtuple("WideHparams", ["field_names", "alpha", "beta", "L1", "L2"])


def _sigmoid(example_idx,logit):
    return 1/(1+np.exp(-logit))

class WideEstimator(BaseEstimator):
    def __init__(self,hparams,data_source):

        self._layer=WideLayer(field_names=hparams.field_names,
                              alpha=hparams.alpha,
                              beta=hparams.beta,
                              L1=hparams.L1,
                              L2=hparams.L2,
                              prob_fn=_sigmoid
                              )

        super().__init__(data_source)

    def train_batch(self,features,labels):
        return self._layer.train(sp_features=features,labels=labels)

    def predict(self,features):
        pred_logits=self._layer.predict_logits(sp_features=features)
        return 1/(1+np.exp(-pred_logits))
