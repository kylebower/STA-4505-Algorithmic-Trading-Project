import numpy as np

class env():
    def __init__(self,pricemin,pricemax,dt=1,dT=60):
        self.dt = dt
        self.dT = dT
        self.state_T = 0
        self.state_q = 0
        self.NT = 10
        self.kappa = 1/dT
        self.theta = 1
        self.sigma = 0.02/np.sqrt(dT)
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
        x = (action - 5)/self.dT
        t = 1
        q0 = self.state_q
        period_reward = 0
        while t < self.dT + 1:
            reward_, q_, s_ = self.SimMRStep(S0=self.state_s, q0=self.state_q, x=x, kappa=self.kappa, theta=self.theta, sigma=self.sigma, dt=self.dt, phi=self.phi)
            self.state_q = q_
            self.state_s = s_
            period_reward += reward_
            t += 1
        if self.state_T < self.NT-1:
            done = False
            self.state_T += 1
            return np.array([self.state_T,  self.state_q, self.state_s]), period_reward, done
        else:
            done = True
            reward_, q_, s_ = self.SimMRStep(S0=self.state_s, q0=self.state_q, x=x, kappa=self.kappa, theta=self.theta, sigma=self.sigma, dt=self.dt, phi=self.phi)
            terminal_reward = self.state_q * (s_ - self.state_s) -self.c * np.square(self.state_q)
            period_reward += terminal_reward
            self.state_s = s_
            self.state_q = 0
            self.state_T += 1
            return np.array([self.state_T,  self.state_q, self.state_s]), period_reward, done



    def SimMRStep(self, S0, q0, x, kappa, theta, sigma, dt, phi):
        S1 = theta + (S0 - theta) * np.exp(-kappa * dt) + sigma * np.sqrt(dt) * np.random.randn()
        q1 = q0 + x
        reward = q1 * (S1 - S0) - phi * np.square(x)
        #reward = q0 * (S1 - S0) - phi * np.square(x)
        return reward, q1, S1