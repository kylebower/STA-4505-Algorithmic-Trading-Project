% Create a Q-Learning Agent

env = rlPredefinedEnv("BasicGridWorld");

qTable = rlTable(getObservationInfo(env),getActionInfo(env));
critic = rlQValueRepresentation(qTable,getObservationInfo(env),getActionInfo(env));

opt = rlQAgentOptions;
opt.EpsilonGreedyExploration.Epsilon = 0.05;

agent = rlQAgent(critic,opt)

getAction(agent,{randi(25)})