import numpy as np


class Adagrad:
    def __init__(self, lr):
        self._lr = lr

        self._sum_grad2 = {}

    def update(self, variables, gradients):
        for gradname, gradient in gradients.items():
            g2 = gradient * gradient
            if gradname in self._sum_grad2:
                self._sum_grad2[gradname] += g2
            else:
                self._sum_grad2[gradname] = g2

            delta = self._lr * gradient / (np.sqrt(self._sum_grad2[gradname]) + 1e-6)

            if '@' in gradname:
                varname, row = gradname.split("@")
                row = int(row)
                variable = variables[varname]
                variable[row, :] -= delta
            else:
                variable = variables[gradname]
                variable -= delta


class SGD:
    def __init__(self, lr):
        self._lr = lr
        self._sum_grad2 = {}

    def update(self, variables, gradients):
        for gradname, gradient in gradients.items():
            g = gradient
            if gradname in self._sum_grad2:
                self._sum_grad2[gradname] += g
            else:
                self._sum_grad2[gradname] = g
            delta = self._lr * g
            if "@" in gradname:
                varname, row = gradname.split("@")
                row=int(row)
                variable=variables[varname]
                variable[row,:]-=delta
            else:
                variable = variables[gradname]
                variable -= delta
