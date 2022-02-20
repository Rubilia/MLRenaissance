import numpy as np


class DiscreteActionSpace(object):
    def __init__(self, action_dimensions: list):
        self.action_depth = len(action_dimensions)
        self.ranges = action_dimensions
        self.total_actions = np.prod(list(map(lambda arr: arr[1] - arr[0] + 1, action_dimensions)))
