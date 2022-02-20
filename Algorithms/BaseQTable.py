class BaseQTable(object):
    def __init__(self, A_space, S_space):
        self.ActionSpace = A_space
        self.StateSpace = S_space
        self.Q_table = []

    def reset(self):
        self.Q_table = []
