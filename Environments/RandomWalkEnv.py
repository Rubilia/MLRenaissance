from abc import ABC

from Algorithms.utils.Spaces import BinaryStateSpace, DiscreteActionSpace
from Environments.baseEnv import BaseDiscreteEnv


class RandomWalkEnv(BaseDiscreteEnv, ABC):
    def __init__(self, n_nodes=10, gamma=.9):
        self.n_nodes = 10
        self.state_space = BinaryStateSpace(n_nodes)
        self.action_space = DiscreteActionSpace([[0, 2]])
        self.position = 0
        self.game_over = False
        self.steps = 0
        self.gamma = gamma

    def getDiscountFactor(self) -> float:
        return self.gamma

    def getCurrentState(self):
        return self.state_space.reverse_state(self.position)

    def getStateSpace(self):
        return self.state_space

    def getActionSpace(self):
        return self.action_space

    def reset(self):
        self.position = 0
        self.game_over = False
        self.steps = 0

    def isTerminated(self) -> bool:
        return self.game_over

    # 0 - left
    # 1 - right
    # 2 - no action
    def step(self, a: DiscreteActionSpace):
        self.steps += 1

        a_decrypted = self.action_space.compress_action(a)
        if self.position == 0 and a_decrypted == 0 or self.steps > 50:
            self.game_over = True
            return self.getCurrentState(), -1.
        elif self.position == self.n_nodes - 1 and a_decrypted == 1:
            self.game_over = True
            return self.getCurrentState(), 5.

        if a_decrypted == 0:
            self.position -= 1
        elif a_decrypted == 1:
            self.position += 1

        return self.getCurrentState(), 0.
