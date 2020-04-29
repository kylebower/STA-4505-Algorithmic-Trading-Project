function [reward, q1, S1]= SimMRStep(S0, q0, x, kappa, theta, sigma, dt)
% S0 - initial stock price
% q0 - initial inventory
% x  - shares to trade
%
% reward - one-step reward
% q1 - terminal inventory
% S1 - terminal stock price


    S1 = theta + (S0-theta)*exp(-kappa*dt) + sigma*sqrt(dt)*randn();

    q1 = q0 + x;
    
    phi =0;
    reward = q1 * (S1-S0) - phi*x^2;


end

