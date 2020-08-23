import numpy as np

def gain_loss_ratio(returns):
    profits = np.array([max(i,0) for i in returns])
    profit = np.sum(profits)
    num_profit = sum([int(i >0) for i in returns])
    losses = np.array([min(i,0) for i in returns])
    loss = np.sum(losses)
    num_loss = np.shape(returns)[0] - num_profit
    avg_profit = profit/num_profit
    avg_loss = loss/num_loss
    gain_loss_ratio = abs(avg_profit/avg_loss)
    return gain_loss_ratio

def var(returns, alpha = 0.01):
    sorted_returns = np.sort(returns)
    index = int(alpha * len(sorted_returns))
    var = sorted_returns[index]
    return var

def lpm(returns, threshold, order):
    thresholds = np.empty(len(returns))
    thresholds.fill(threshold)
    diffs = (thresholds - returns).clip(min=0)
    return np.sum(diffs ** order) / len(returns)

def hpm(returns, threshold, order):
    thresholds = np.empty(len(returns))
    thresholds.fill(threshold)
    diffs = (returns - thresholds).clip(min=0)
    return np.sum(diffs ** order) / len(returns)

def omega_ratio(returns, risk_free_rate, target=0):
    omega_ratio = (np.mean(returns) - risk_free_rate)/ lpm(returns, target, 1)
    return omega_ratio

def sortino_ratio(returns, risk_free_rate, target=0):
    sortino_ratio = (np.mean(returns) - risk_free_rate) / np.sqrt(lpm(returns, target, 2))
    return sortino_ratio


def tvar(returns, alpha= 0.01):
    sorted_returns = np.sort(returns)
    index = int(alpha * len(sorted_returns))
    #sum all returns under var
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    tvar = sum_var / index
    return tvar

def summary(returns, alpha, risk_free_rate, target):
    print('Strategy has VaR at level ' + str(alpha) + ' is ' + str(np.round(var(returns,alpha),3)))
    print('Strategy has TVaR at level ' + str(alpha) + ' is ' + str(np.round(tvar(returns,alpha),3)))
    print('Strategy has gain loss ratio: '+ str(np.round(gain_loss_ratio(returns), 3)))
    print('Strategy has omega ratio ' + str(np.round(omega_ratio(returns,risk_free_rate, target), 3)))
    print('Strategy has Sortino ratio ' + str(np.round(sortino_ratio(returns, risk_free_rate, target), 3)))


