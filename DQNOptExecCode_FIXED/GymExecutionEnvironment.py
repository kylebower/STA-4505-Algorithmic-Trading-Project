import numpy as np

class env():
    def __init__(self,pricemin,pricemax,dt=1):
        self.dt = dt
        self.state_T = 0
        self.state_q = 0
        self.NT = 10
        self.kappa = 1
        self.theta = 1
        self.sigma = 0.02
        self.phi = 0.000001
        self.c = 0.01
        self.smin = self.theta - 3 * self.sigma / np.sqrt(2 * self.kappa)  # min price
        self.smax = self.theta + 3 * self.sigma / np.sqrt(2 * self.kappa)  # max price
        self.state_s =  np.random.uniform(low=self.smin, high=self.smax)


    def reset(self):
        self.reward = 0
        self.state_s = np.random.uniform(low=self.smin, high=self.smax)
        self.state_q = 0
        self.state_T = 0
        return np.array([self.state_T, self.state_q, self.state_s])

    def step(self, action):
        x = action

        reward_, q_, s_ = self.SimMRStep(S0=self.state_s, q0=self.state_q, x=x-5, kappa=self.kappa, theta=self.theta, sigma=self.sigma, dt=self.dt, phi=self.phi)

        self.state_q = q_
        self.state_s = s_

        if self.state_T < self.NT- 1:
            done = False
            self.state_T += 1
            return np.array([self.state_T, q_, s_]), reward_, done
        else:
            done = True
            terminal_reward = - self.c * np.square(q_)
            reward_ += terminal_reward
            self.state_T += 1
            return np.array([self.state_T, q_, s_]), reward_, done

    def SimMRStep(self, S0, q0, x, kappa, theta, sigma, dt, phi):
        S1 = theta + (S0 - theta) * np.exp(-kappa * dt) + sigma * np.sqrt(dt) * np.random.randn()
        q1 = q0 + x
        reward = q1 * (S1 - S0) - phi * np.square(x)
        #reward = q0 * (S1 - S0) - phi * np.square(x)
        return reward, q1, S1
