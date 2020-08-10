import gym
from DDQNetwork_OptimalExecution import Agent
from utils import plot_learning_curve
import numpy as np
import GymExecutionEnvironment as EEnv
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch as T

if __name__ == '__main__':
    env = EEnv.env(pricemin=0.8, pricemax=1.2, dt=1)
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, n_actions=11, input_dims=[3], mem_size=10000,
                  batch_size=64, eps_min=0.01, eps_dec=5e-6, replace=1000, fc1_dims=64,
                  fc2_dims=64)
    scores, eps_history = [], []
    n_games = 10

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i + 1, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    x = [i + 1 for i in range(n_games)]
    filename = 'opt_exec_ddqn.png'
    plot_learning_curve(x, scores, eps_history, filename)



    def get_optimal_actions(i):
        Q = np.zeros([21, 21])
        for j in range(21):
            for k in range(21):
                observation = T.tensor([float(i), float(k) - 10, 0.8 + (float(j) - 1) * 0.4 / 21], dtype=T.float)
                q_eval = agent.q_eval.forward(observation)
                Q[k, j] = q_eval.argmax() - 5
        return Q

    plt.clf()
    # xticklabels = np.arange(-10, 11, 1)
    # yticklabels = np.round(np.arange(0.8, 1.2, 0.4 / 21), 3)
    # yticks = np.round(np.arange(0.8, 1.2, 0.1), 1)
    cmap = sns.diverging_palette(20, 240, as_cmap=True)  # husl color system

    for i in range(11):
        FONT_SIZE = 14
        plt.rc('axes', labelsize=FONT_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=FONT_SIZE)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=FONT_SIZE)  # fontsize of the y tick labels

        num_xticks = 5
        num_yticks = 5
        data = get_optimal_actions(i)
        xticks = np.linspace(0, len(data) - 1, num_xticks, dtype=np.int)
        yticks = np.linspace(0, len(data) - 1, num_yticks, dtype=np.int)
        xticklabels = xticks - 10
        smin = 0.8
        smax = 1.2
        yticklabels = np.round(yticks * (smax-smin) / (len(data)-1) + smin, 2)

        heat_map = sns.heatmap(data, cmap=cmap, xticklabels=xticklabels,  yticklabels=yticklabels,  vmin=-5, vmax=5)
        heat_map.set_xticks(xticks)
        heat_map.set_yticks(yticks)

        heat_map.invert_yaxis()
        plt.xlabel("Inventory")
        plt.ylabel("Price")

        plt.show()

