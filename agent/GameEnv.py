import random
import numpy as np

class ActionSpace:
    """docstring for ActionSpace"""
    def __init__(self, n):
        self.n = n

    def sample(self):
        result = []
        for i in range(self.n):
            result.append(random.random()-0.5)
        return result #random.randint(0, self.n-1)

class ObservationSpace:
    def __init__(self, shape):
        self.shape = (shape,)