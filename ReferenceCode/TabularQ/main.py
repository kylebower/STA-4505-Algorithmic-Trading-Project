from q_learning import q_learning, plot_actions
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch as T

if __name__ == '__main__':
    q_table, s_matrix, q_matrix, a_matrix, r_matrix, pnls = q_learning()
    np.save("q.npy", q_table)
    q_table_ = np.load("q.npy")
    print((q_table_ == q_table).all())

    plt.hist(q_matrix[:-1, -1])
    plt.ylabel("frequency")
    plt.show()
    plot_actions(1,q_table_)
    plot_actions(3,q_table_)

    sns.distplot(pnls[-100:], norm_hist=True)
    plt.ylabel("frequency")
    plt.xlabel("pnl")
    plt.show()
    print("Mean of Pnl:")
    print(np.mean(pnls[-100:]))
