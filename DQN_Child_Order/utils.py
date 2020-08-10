import matplotlib.pyplot as plt
import numpy as np
import gym
import torch as T
import seaborn as sns

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def get_optimal_actions(time,smin,smax,invgrid,pricegrid,actiongrid,agent):
    Q = np.zeros([pricegrid, invgrid])
    for j in range(pricegrid):
        for k in range(invgrid):
            observation = T.tensor([float(time), float(k - (invgrid-1)/2), smin + (float(j) - 1) * (smax-smin) / pricegrid], dtype=T.float).to(agent.q_eval.device)
            actions = agent.q_eval.forward(observation).detach().numpy()
            lowerbound = max(-5, int(-(invgrid-1)/2 - (k - (invgrid-1)/2))) + int((actiongrid-1)/2)
            upperbound = min(5, int((invgrid-1)/2 - (k - (invgrid-1)/2))) + int((actiongrid-1)/2)
            for i in range(lowerbound, upperbound + 1):
                if i == lowerbound:
                    maxactionval = actions[i]
                    action = i
                else:
                    if actions[i] > maxactionval:
                        maxactionval = actions[i]
                        action = i
            Q[j,k] = action - int((actiongrid-1)/2)
    return Q


def plot_heatmap(t,smin,smax,invgrid,pricegrid,actiongrid,agent):
    yticklabels = np.round(np.arange(smin, smax, (smax - smin) / pricegrid), 3)
    xticklabels = np.arange(-int((invgrid-1)/2), int((invgrid-1)/2)+1, 1)
    cmap = sns.diverging_palette(20, 240, as_cmap=True)  # husl color system
    heat_map = sns.heatmap(get_optimal_actions(t,smin,smax,invgrid,pricegrid,actiongrid,agent), cmap=cmap, yticklabels=yticklabels, xticklabels=xticklabels,
                           vmin=-int((actiongrid-1)/2), vmax=int((actiongrid-1)/2))
    heat_map.invert_yaxis()
    plt.xlabel("Inventory")
    plt.ylabel("Price")
    filename = 'opt_exec_ddqn_optaction_time' + str(t) + '.png'
    plt.savefig(filename)
    plt.show()
