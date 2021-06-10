from tqdm import tqdm
import numpy as np

def epsilon_greedy(env, epsilon=0.1, initial_value=0.0, steps=1000):
    """ solve env with ε-greedy (sample-average) algorithm """
    k_q_evals = np.full(env.k, initial_value, dtype=np.float64)         # initial values of Q(a)
    k_times = np.zeros(env.k)                                           # selected times of each a

    rewards = []
    for t in range(steps):
        # random selection
        if np.random.rand() < epsilon: a = np.random.randint(env.k)
        # greedy selection
        else: a = np.argmax(k_q_evals)

        cur_reward = env.step(a)
        k_times[a] += 1
        k_q_evals[a] += (cur_reward-k_q_evals[a])/k_times[a]            # update Q(a)
        rewards.append(cur_reward)
    return rewards

def upper_confidence_bound(env, c=1, steps=1000):
    """ solve env with Upper-Confidence-Bound action selection """
    k_q_evals = np.zeros(env.k, dtype=np.float64)                       # initial values of Q(a)
    k_times = np.zeros(env.k)                                           # selected times of each a

    rewards = []
    for t in range(steps):
        a = np.argmax(k_q_evals+c*np.sqrt(np.log(t+1)/(k_times+0.01)))  # action value + uncertainty
        cur_reward = env.step(a)
        k_times[a] += 1
        k_q_evals[a] += (cur_reward-k_q_evals[a])/k_times[a]            # update Q(a)
        rewards.append(cur_reward)
    return rewards

def gradient_bandit(env, alpha=0.1, steps=1000):
    """ solve env with gradient bandit algorithm """
    k_prefers = np.zeros(env.k, dtype=np.float64)                       # initial numerical preference of actions

    rewards = []
    for t in range(steps):
        exp_prefers = np.exp(k_prefers)
        probs = exp_prefers/np.sum(exp_prefers)                         # action probabilities
        a = np.random.choice(np.arange(env.k), p=probs)                   # action choice

        cur_reward = env.step(a)
        average_reward = cur_reward if len(rewards) == 0 else np.mean(rewards)
        k_prefers += np.array([alpha*(cur_reward-average_reward)*(int(i==a)-probs[i]) for i in range(env.k)])
        rewards.append(cur_reward)
    return rewards