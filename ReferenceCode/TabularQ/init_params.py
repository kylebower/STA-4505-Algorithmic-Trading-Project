import numpy as np
# global variables
epsilon_a = 10000  # greedy policy is epsilon_a/(epsilon_b+k)
epsilon_b = 10000
alpha_a = 100000  # learning rate is alpha_a/(alpha_b+k)
alpha_b = 100000
gamma = 0.999  # discount factor
niter = int(1e6)  # number of iterations
NT = int(10)  # 10 periods
dt = 1  # send child order each dt
dT = 60  # decisions are made at each dT

kappa = 1/60  # second time scale: order 1/dT
theta = 1
sigma = 0.02/np.sqrt(60)  # order 1/sqrt(dT)
phi = 0.000001
c = 1000

Qmax = 10  # max inventory
Qmin = -10  # min inventory
q_grid = list(range(Qmin, Qmax + 1))

a_grid = list(range(-5, 6))  # actions
# a_grid.reverse
a_grid.remove(0)
a_grid = [0] + a_grid

s_min = theta - 3 * sigma / np.sqrt(2 * kappa)  # min price
s_max = theta + 3 * sigma / np.sqrt(2 * kappa)  # max price
Ns = 21  # number of prices
ds = (s_max - s_min) / (Ns - 1)
s_grid = np.arange(s_min, s_max + ds / 2, ds).tolist()
