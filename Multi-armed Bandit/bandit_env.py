from tqdm import tqdm
import numpy as np

class BanditEnv(object):
    """ Environment of k-armed Bandit """

    def __init__(self, k):
        """ init Bandit environment """
        self.k = k                              # number of levers, parameter of k-armed Bandit
        self.k_qs = np.random.randn(k)          # k expected rewards of each levers

    def step(self, action):
        """ step action (0, 1, ..., or k-1) to env """
        return np.random.normal(self.k_qs[action], 1)   # return reward

if __name__ == '__main__':
    """ test BanditEnv """

    env = BanditEnv(k=10)
    print("[#] Initialized %d-armed Bandit environment."%env.k)
    print("[#] True values of each levers: %ls"%env.k_qs)

    k_q_evals = np.zeros(env.k)                 # initial values of Q(a)
    k_times = np.zeros(env.k)                   # selected times of each a
    for it in tqdm(range(100000)):
        a = np.random.randint(10)               # random policy
        cur_reward = env.step(a)
        k_times[a] += 1
        k_q_evals[a] += (cur_reward-k_q_evals[a])/k_times[a]

    print("[#] Estimated values of each levers: %ls"%k_q_evals)
    print("[#] The average error is: %f"%np.mean(np.abs(k_q_evals-env.k_qs)))
