from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from bandit_env import BanditEnv
from bandit_algorithms import *

def main():
    """ Test 4 algorithms (ε-greedy, Upper Confidence Bound, gradient bandit
    and greedy with optimistic initial values) on 10-armed bandit testbed.
    """
    print("[#] Test algorithms: ε-greedy, greedy with optimistic initial values, Upper-Confidence-Bound, gradient bandit.")
    print("[#] Test environment: 10-armed bandit.\n")
    k = 10

    print("[#] Parameter study for 4 algorithms.")
    fig, ax = plt.subplots()
    ax.set_xscale('log', basex=2)

    # ε-greedy
    print("[1] Test ε-greedy with ε ranging from 0.01 to 0.25.")
    epsilons = np.arange(0.01, 0.26, 0.01)
    average_rewards = np.zeros_like(epsilons)
    for i in tqdm(range(len(epsilons))):
        # 10000 tests
        rewards = [epsilon_greedy(BanditEnv(10), epsilon=epsilons[i]) for it in range(10000)]
        average_rewards[i] = np.mean(rewards)
    best_epsilon = epsilons[np.argmax(average_rewards)]
    print("[1] Best ε for ε-greedy method is %.2f.\n"%best_epsilon)
    ax.plot(epsilons, average_rewards, label=r'$\epsilon$-greedy ($\epsilon$)')

    # greedy method with optimistic initial values
    print("[2] Test greedy method with initial Q(a) ranging from 0.25 to 4.")
    Q_inits = np.arange(0.25, 4, 0.1)
    average_rewards = np.zeros_like(Q_inits)
    for i in tqdm(range(len(Q_inits))):
        # 10000 tests
        rewards = [epsilon_greedy(BanditEnv(10), epsilon=0, initial_value=Q_inits[i]) for it in range(10000)]
        average_rewards[i] = np.mean(rewards)
    best_Qinit = Q_inits[np.argmax(average_rewards)]
    print("[2] Best initial Q(a) for greedy method is %.2f.\n"%best_Qinit)
    ax.plot(Q_inits, average_rewards, label=r'greedy with optimistic initialization ($Q_0$)')

    # Upper Confidence Bound
    print("[3] Test Upper-Confidence-Bound action selection with c(weight of uncertainty) ranging from 0.05 to 4.")
    cs = np.arange(0.05, 4, 0.1)
    average_rewards = np.zeros_like(cs)
    for i in tqdm(range(len(cs))):
        # 10000 tests
        rewards = [upper_confidence_bound(BanditEnv(10), c=cs[i]) for it in range(10000)]
        average_rewards[i] = np.mean(rewards)
    best_c = cs[np.argmax(average_rewards)]
    print("[3] Best c for Upper-Confidence-Bound action selection is %.2f.\n"%best_c)
    ax.plot(cs, average_rewards, label='Upper Confidence Bound (c)')

    # gradient bandit
    print("[4] Test gradient bandit algorithm with α(learning rate) ranging from 0.02 to 4.")
    alphas = np.arange(0.02, 4, 0.1)
    average_rewards = np.zeros_like(alphas)
    for i in tqdm(range(len(alphas))):
        # 1000 tests
        rewards = [gradient_bandit(BanditEnv(10), alpha=alphas[i]) for it in range(1000)]
        average_rewards[i] = np.mean(rewards)
    best_alpha = alphas[np.argmax(average_rewards)]
    print("[4] Best α for gradient bandit algorithm is %.2f.\n"%best_alpha)
    ax.plot(alphas, average_rewards, label=r'gradient bandit ($\alpha$)')

    print("[#] Plot the average reward (over first 1000 steps) of 4 algorithms under different parameters respectively.")
    plt.axis([0, 4, 1, 1.6])
    plt.xlabel(r'$\epsilon$    $\alpha$    c    $Q_0$')
    plt.ylabel('Average reward over first 1000 steps')
    plt.legend()
    plt.savefig("./figs/parameter_study.png", dpi=500)

    # 2000 runs for best parameter
    print("[#] Test 4 algorithms with best parameters respectively.")
    print("[#] Each algorithm runs for 1000 steps in one test.")
    print("[#] 2000 runs for each algorithm to calculate average reward at each step.")
    epsilon_greedy_rewards = []
    greedy_optimistic_init_rewards = []
    UCB_rewards = []
    gradient_bandit_rewards = []
    steps, runs= 1000, 2000
    for it in tqdm(range(runs)):
        epsilon_greedy_rewards.append(epsilon_greedy(BanditEnv(10), epsilon=best_epsilon))
        greedy_optimistic_init_rewards.append(epsilon_greedy(BanditEnv(10), epsilon=0, initial_value=best_Qinit))
        UCB_rewards.append(upper_confidence_bound(BanditEnv(10), c=best_c))
        gradient_bandit_rewards.append(gradient_bandit(BanditEnv(10), alpha=best_alpha))

    print("[#] Plot average reward (over 2000 runs) v.s. steps of 4 algorithms with best parameters respectively.")
    fig, ax = plt.subplots()
    ax.plot(np.mean(epsilon_greedy_rewards, axis=0), label=r'$\epsilon$-greedy ($\epsilon$=%.2f)'%best_epsilon)
    ax.plot(np.mean(greedy_optimistic_init_rewards, axis=0), label=r'greedy with optimistic initialization ($Q_0$=%.2f)'%best_Qinit)
    ax.plot(np.mean(UCB_rewards, axis=0), label='Upper Confidence Bound (c=%.2f)'%best_c)
    ax.plot(np.mean(gradient_bandit_rewards, axis=0), label=r'gradient bandit ($\alpha=%.2f$)'%best_alpha)

    plt.xlabel('steps')
    plt.ylabel('Average reward over 2000 runs')
    plt.legend()
    plt.savefig("./figs/comparison.png", dpi=500)
    plt.show()

if __name__ == '__main__': main()
