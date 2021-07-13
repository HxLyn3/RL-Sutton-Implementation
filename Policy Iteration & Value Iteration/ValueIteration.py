import copy
import numpy as np

class ValueIteration():
    """ an implementation of Value-Iteration method """

    def __init__(self, env, gamma=1):
        self.env = env
        self.gamma = gamma
        self.state_space = self.env.enumerate_state()
        # action space: 0 -- up, 1 -- down, 2 -- left, 3 -- right
        self.nactions = self.env.nactions
        self.action_space = np.arange(self.nactions)
        self.reset()

    def value_iteration(self):
        """ value iteration """
        delta = 100
        while delta != 0:
            old_values = copy.deepcopy(self.values)
            self.value_iteration_step()
            delta = np.max(np.abs(old_values-self.values))

    def value_iteration_step(self):
        """ one step of Value-Iteration """
        for state in self.state_space:
            if self.env.isDone(state): continue
            action_values = []
            for action in self.action_space:
                next_state, reward, done = self.env.transition(state, action)
                next_state_value = self.values[next_state[1]][next_state[0]]
                action_values.append(reward+self.gamma*next_state_value)
            max_action_value = np.max(action_values)
            self.policy[state[1]][state[0]] = (action_values==max_action_value).astype(np.float32)
            self.policy[state[1]][state[0]] /= np.sum(self.policy[state[1]][state[0]])
            self.values[state[1]][state[0]] = max_action_value

    def reset(self):
        """ reset values and policy """
        self.values = np.zeros(shape=(self.env.nrows, self.env.ncols))
        self.policy = np.ones(shape=(self.env.nrows, self.env.ncols, self.nactions))/self.nactions
        for state in self.state_space:
            if self.env.isDone(state): self.policy[state[1]][state[0]] = np.zeros(self.nactions)
