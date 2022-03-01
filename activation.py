import numpy as np


class sigmoid:
    def __init__(self):
        self._last_forward_result = None

    def forward(self, X):
        self._last_forward_result = 1.0 / (1.0 + np.exp(-X))
        return self._last_forward_result

    def backward(self, prev_grads):
        assert prev_grads.shape == self._last_forward_result.shape

        return prev_grads * self._last_forward_result * (1 - self._last_forward_result)


class ReLU:
    def __init__(self):
        self._last_input = None

    def forward(self, X):
        self._last_input = X
        return np.maximum(0, X)

    def backward(self, pred_grads):
        assert pred_grads.shape == self._last_input.shape
        local_grads = np.zeros_like(self._last_input)
        local_grads[self._last_input > 0] = 1.0
        return pred_grads * local_grads
