import random

import numpy as np

from Algorithms.utils.Spaces import DiscreteActionSpace, DiscreteStateSpace
from Environments.RandomWalkEnv import RandomWalkEnv
from Environments.baseEnv import BaseDiscreteEnv


class BaseQTable(object):
    def __init__(self, A_space, S_space, gamma=.9):
        self.ActionSpace: DiscreteActionSpace = A_space
        self.StateSpace: DiscreteStateSpace = S_space
        self.init_table()
        self.env: BaseDiscreteEnv = None
        self.gamma = gamma

    def init_table(self):
        self.Q_table = {}
        for a in range(self.ActionSpace.total_actions):
            self.Q_table[a] = {}

    def reset(self):
        self.init_table()
        self.env.reset()

    def softmax(self, vals: list):
        den = sum(list(map(np.exp, vals)))
        return list(map(lambda x: np.exp(x) / den, vals))

    def choose_action(self, s):
        s_compressed = self.StateSpace.compress_state(s)
        if not s_compressed in self.Q_table[0].keys():
            for a in range(self.ActionSpace.total_actions):
                self.Q_table[a][s_compressed] = 0.

        Q_a = [self.Q_table[a][s_compressed] for a in range(self.ActionSpace.total_actions)]
        return random.choices(population=list(range(self.ActionSpace.total_actions)), weights=self.softmax(Q_a), k=1)[0]

    def set_env(self, env: BaseDiscreteEnv):
        self.env = env

    def train(self, iterations=16):
        total_r = 0
        exp = []
        for t in range(iterations):
            s = self.env.getCurrentState()
            a = self.choose_action(s)
            new_state, r = self.env.step(self.ActionSpace.reverse_action(a))
            total_r += r

            exp += [[s, a, new_state, r]]

            if self.env.isTerminated():
                self.env.reset()

        self.choose_action(new_state)

        # Training loop
        for (s, a, new_state, r) in exp:
            self.Q_table[a][self.StateSpace.compress_state(s)] = r + self.gamma * max([self.Q_table[a_][self.StateSpace.compress_state(new_state)] for a_ in range(self.ActionSpace.total_actions)])

        return total_r

    def train_epoch(self, epochs=100):
        for t in range(epochs):
            r = self.train()
            print(f'Iteration #{t}: total reward over the training epoch is: {r}')


if __name__ == '__main__':
    env = RandomWalkEnv(n_nodes=10)

    agent = BaseQTable(env.action_space, env.state_space, gamma=env.gamma)
    agent.set_env(env)
    agent.train_epoch(100)
