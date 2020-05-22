% Q-Learning with Epsilon-Greedy Policy

nIterations = 1;

gamma = 0.5; % discount factor
X_0 = 1; % initial price

kappa = 1; % second time scale
theta = 1;
sigma = 0.02;
phi = 0;
c = 0;

N = 5;
T = 5;

for k = 1:nIterations
    alpha_k = 1/k; % Learning rate
    epsilon_k = 1/k; % exploration parameter
    X_max = theta + 5*sigma/sqrt(2*kappa);
    X_min = theta + 5*sigma/sqrt(2*kappa);
    
    for j=0:N
        t = j*T/N;
    end
end

