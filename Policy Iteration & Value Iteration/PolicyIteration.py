import copy
import numpy as np

class PolicyIteration():
    """ an implementation of Policy-Iteration method """

    def __init__(self, env, gamma=1):
        self.env = env
        self.gamma = gamma
        self.state_space = self.env.enumerate_state()
        # action space: 0 -- up, 1 -- down, 2 -- left, 3 -- right
        self.nactions = self.env.nactions
        self.action_space = np.arange(self.nactions)
        self.reset()

    def policy_evaluation(self):
        """ policy evaluation """
        delta = 100
        while delta > 1e-4:
            old_values = copy.deepcopy(self.values)
            for state in self.state_space:
                if self.env.isDone(state): continue
                state_value = 0
                for action in self.action_space:
                    next_state, reward, done = self.env.transition(state, action)
                    next_state_value = self.values[next_state[1]][next_state[0]]
                    state_value += self.policy[state[1]][state[0]][action]*(reward+self.gamma*next_state_value)
                self.values[state[1]][state[0]] = state_value
            delta = np.max(np.abs(old_values-self.values))

    def policy_improvement(self):
        """ policy improvement """
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

    def policy_iteration(self):
        """ policy iteration """
        delta = 100
        while delta != 0:
            self.policy_evaluation()
            old_policy = copy.deepcopy(self.policy)
            self.policy_improvement()
            delta = np.max(np.abs(old_policy-self.policy))

    def reset(self):
        """ reset values and policy """
        self.values = np.zeros(shape=(self.env.nrows, self.env.ncols))
        self.policy = np.ones(shape=(self.env.nrows, self.env.ncols, self.nactions))/self.nactions
        for state in self.state_space:
            if self.env.isDone(state): self.policy[state[1]][state[0]] = np.zeros(self.nactions)
