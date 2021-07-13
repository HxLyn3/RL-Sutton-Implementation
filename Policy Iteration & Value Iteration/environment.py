import copy
import numpy as np

class GridEnv():
    """ a class of Grid-World environment 
        (RxC, destination: (0, 0)/(C-1, R-1), no obstacles) """

    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.nactions = 4   # up, down, left, right
        self.reset()

    def reset(self):
        """ set initial position in Grid-World """
        self.pos = np.array([np.random.randint(self.ncols), np.random.randint(self.nrows)])
        while self.isDone(self.pos):
            self.pos = np.array([np.random.randint(self.ncols), np.random.randint(self.nrows)])

    def get_state(self):
        """ return position """
        return self.pos

    def enumerate_state(self):
        """ return all possible states """
        xs, ys = np.arange(self.ncols), np.arange(self.nrows)
        xs, ys = np.meshgrid(xs, ys)
        states = np.concatenate((xs.reshape(-1, 1), ys.reshape(-1, 1)), axis=1)
        return states

    def isDone(self, state):
        """ done = True when reached destination """
        return (state == np.array([0, 0])).all() or \
               (state == np.array([self.ncols-1, self.nrows-1])).all()

    def transition(self, state, action):
        """ action: 0 -- up, 1 -- down, 2 -- left, 3 -- right """
        if self.isDone(state): return None
        next_state = copy.deepcopy(state)
        if action == 0 and next_state[1] > 0: next_state[1] -= 1
        elif action == 1 and next_state[1] < self.nrows - 1: next_state[1] += 1
        elif action == 2 and next_state[0] > 0: next_state[0] -= 1
        elif action == 3 and next_state[0] < self.ncols - 1: next_state[0] += 1
        return next_state, -1, self.isDone(next_state)  # reward = -1

    def step(self, action):
        self.pos, reward, done = self.transition(self.pos, action)
        return self.pos, reward, done
