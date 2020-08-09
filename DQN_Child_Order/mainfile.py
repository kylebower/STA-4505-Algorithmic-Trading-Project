from DDQNetwork_OptimalExecutionV2 import Agent
from utils import plot_learning_curve
from utils import plot_heatmap
import numpy as np
import GymExecutionEnvironment as EEnv
import seaborn as sns
import torch as T
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = EEnv.env(pricemin=0.8, pricemax=1.2, dt=1, dT=60)
    agent = Agent(gamma=0.999, epsilon=1.0, lr=0.001, n_actions=11, input_dims=[3], mem_size=25000,
                  batch_size=64, eps_min=0.05, eps_dec=1e-6, replace=5000, fc1_dims=64,
                  fc2_dims=64)
    n_games = 100000

    scores, avg_score, eps_history = agent.train(env, n_games=n_games)

    invgrid = 21
    pricegrid = 21
    actiongrid = 11

    for i in range(env.NT + 1):
        plot_heatmap(i, env.smin, env.smax, invgrid, pricegrid, actiongrid, agent)

    x = [i + 1 for i in range(n_games)]
    filename = 'opt_exec_ddqn_learningcurve.png'
    plot_learning_curve(x, scores, eps_history, filename)
    agent.save_model()