kappa = 1; % second time scale
theta = 1;
sigma = 0.02;
phi = 0;
c = 0;

iter = 1e7;


% Q grid
Qmax = 10;
Qmin = -10;
q_grid = [Qmin:Qmax]';
Nq = length(q_grid);

% allowed actions
a_grid = int32([-5:5])';
%a_grid = int32(q_grid);
Na = length(a_grid);

% asset price grid
S_min = theta - 5*sigma/sqrt(2*kappa);
S_max = theta + 5*sigma/sqrt(2*kappa);
NS = 51;
dS = (S_max-S_min)/(NS-1);
S_grid = [S_min : dS : S_max]';
NS = length(S_grid);



% period grid
NT = 10;
T = [0:NT]';

% send child orders each second within each period
dT = 60; % seconds
dt = 1; % seconds

S = NaN(iter,NT+1);
R = NaN(iter,NT+1);
q = NaN(iter,NT+1);
a = NaN(iter,NT+1);



% rewards internal time step
R(i,j) =  ( S(i,j+1) - S(i,j) ) * (q(i,j) + a(i,j) ) ...
        - c * a(i,j)^2;
    
% reward at the teminal time step
R(i,j+1) =  ( S(i,j+2) - S(i,j+1) ) * (q(i,j+1) + a(i,j+1) ) ...
          - 10*c * a(i,j+1)^2;    
