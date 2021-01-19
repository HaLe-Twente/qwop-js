import random

class ActionSpace:
    """docstring for ActionSpace"""
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randint(0, self.n-1)

class ObservationSpace:
    def __init__(self, shape):
        self.shape = (shape,)