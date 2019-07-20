import math


class Var:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.grad_value = None

    def grad(self):
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad() for weight, var in self.children)
        return self.grad_value

    def 