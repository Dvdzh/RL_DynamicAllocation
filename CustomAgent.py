import numpy as np


class CustomAgent():
    def __init__(self, env):
        self.B = env.param["B"]
        self.K = env.param["K"]
        self.q_table = self.create_qtable()
        self.dictio = self.create_dictionnaire(self.B, self.K)[1]

    def create_qtable(self):
        return np.zeros(((self.B+1)*(self.K+1), self.K+1))

    def create_dictionnaire(self, B, K):
        d_indice = []
        d_inv = {}
        c = 0
        for i in range(B+1):
            for j in range(K+1):
                d_indice.append((i, j))
                d_inv[(i, j)] = c
                c += 1
        return d_indice, d_inv
