import numpy as np


class DiscreteActionSpace(object):
    def __init__(self, action_dimensions: list):
        self.action_depth = len(action_dimensions)
        self.ranges = action_dimensions
        self.total_actions = int(np.prod(list(map(lambda arr: arr[1] - arr[0] + 1, action_dimensions))))

    def compress_action(self, a: np.array) -> int:
        action = 0
        counter = np.shape(a)[0] - 1
        current_range = 0
        for item in a[::-1]:
            if counter == np.shape(a)[0] - 1:
                action = item - self.ranges[counter][0]
                current_range = self.ranges[counter][1] - self.ranges[counter][0] + 1
            else:
                action = (item - self.ranges[counter][0]) * current_range + action
                current_range *= self.ranges[counter][1] - self.ranges[counter][0] + 1

            counter -= 1
        return action

    def reverse_action(self, a: int):
        pro_action = []
        current_range = self.total_actions
        counter = 0
        while a > 0:
            current_range //= (self.ranges[counter][1] - self.ranges[counter][0] + 1)
            pro_action += [a // current_range]
            a %= current_range
            counter += 1

        return np.array(pro_action)


class DiscreteStateSpace(object):
    def __init__(self, action_dimensions: list):
        self.state_depth = len(action_dimensions)
        self.ranges = action_dimensions
        self.total_states = np.prod(list(map(lambda arr: arr[1] - arr[0] + 1, action_dimensions)))

    def compress_state(self, a: np.array) -> int:
        state = 0
        counter = np.shape(a)[0] - 1
        current_range = 0
        for item in a[::-1]:
            if counter == np.shape(a)[0] - 1:
                state = item - self.ranges[counter][0]
                current_range = self.ranges[counter][1] - self.ranges[counter][0] + 1
            else:
                state = (item - self.ranges[counter][0]) * current_range + state
                current_range *= self.ranges[counter][1] - self.ranges[counter][0] + 1

            counter -= 1
        return state

    def reverse_state(self, s: int):
        pro_state = []
        current_range = self.total_states
        counter = 0
        while counter < self.state_depth:
            current_range //= (self.ranges[counter][1] - self.ranges[counter][0] + 1)
            pro_state += [s // current_range]
            s %= current_range
            counter += 1

        return pro_state


class BinaryStateSpace(DiscreteStateSpace):
    def __init__(self, dim: int):
        super().__init__([[0, 1]] * dim)
        self.dim = dim


if __name__ == '__main__':
    BState = BinaryStateSpace(10)

    print(BState.reverse_state(100))
