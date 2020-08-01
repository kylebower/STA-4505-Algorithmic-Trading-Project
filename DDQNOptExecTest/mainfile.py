import gym
from DDQNetwork_OptimalExecution import Agent
from utils import plot_learning_curve
import numpy as np
import GymExecutionEnvironment as EEnv

if __name__ == '__main__':
    env = EEnv.env(pricemin=0.8,pricemax=1.2,dt=1)
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, n_actions=11, input_dims=[3], mem_size=10000,
                  batch_size=64, eps_min = 0.01, eps_dec = 5e-6, replace=1000, fc1_dims=64,
                  fc2_dims=64)
    scores, eps_history = [], []
    n_games = 20000

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done  = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i+1 , 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    x = [i + 1 for i in range(n_games)]
    filename = 'opt_exec_ddqn.png'
    plot_learning_curve(x, scores, eps_history, filename)
