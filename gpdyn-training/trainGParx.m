 function [hyp, flogtheta, i] = trainGParx(hyp, inf, mean, cov, lik, input, target, minf, iters)
% Function for the optimisation (training) of GP model hyperparameters
% 
%% Syntax
%  [hyp, flogtheta, i] = trainGParx(hyp, inf, mean, cov, lik, input, target)
% 
%% Description
% Function for the optimisation (training) of GP model hyperparameters
% based on the training data via maximum marginal likelihood. 
% Uses routines gp and minimize.
% Based on the work of C.E.Rasmussen. 
% 
% Input:
% * hyp    ... the structure of initial hyperparameters
% * inf    ... the function specifying the inference method 
% * cov    ... the prior covariance function (see below)
% * mean   ... the prior mean function
% * lik    ... the likelihood function
% * input  ... the n by D matrix of training inputs
% * target ... the column vector of length n of training targets
% * minf   ... the function handle of the minimization method to be used 
%                (optional, default=@minimize)
% * iters  ... the number of iterations (length) given to the minimizer
%               (optional, default = -100); see minimize(). Note that the
%               minimizer will be called multiple times, each time with the
%               given iters length, until it fails to improve the solutions
%
% Output:
% * hyp       ... optimized hyperparameters 
% * flogtheta ... the minus log likelihood for the different runs (init. to 0)
% * i         ... the number of iterations needed for the last optimization
%
% See Also:
% gp, minimize, trainlgmp, covFunctions, infMethods, likFunctions,
% meanFunctions 
% 
% Examples:
% demo_example_gp_training
%
%%

if (nargin < 8) || isempty(minf)
    minf = @minimize;
end

if nargin < 9
    iters = -100;
end

% [n, D] = size(input);
  
MIN_DIFF = 0.000001; %0.002; 
flogtheta = 0; 

[hyp, flogthetatmp,i] = feval(minf, hyp, @gp, iters, inf, mean, cov, lik, input, target);


if isempty(flogthetatmp)
    flogtheta = flogthetatmp;
    error('minimization failed - matrix close to singular');
else
    flogtheta = [flogtheta flogthetatmp(end)];
end

curIter = 0;
MAXITER = 100;
ALPHA = 1.1;  % After each iteration below, the threshold MIN_DIFF is increased by this factor, to shorten the run

while (curIter < MAXITER && abs(1 - flogtheta(end-1)/flogtheta(end))>MIN_DIFF && abs(flogtheta(end) - flogtheta(end-1))>MIN_DIFF)
    curIter = curIter + 1;
    MIN_DIFF = MIN_DIFF * ALPHA;  % increase the threshold slightly
    
    fprintf('Iteration #%d\n', curIter); 
    disp(strcat(['delta flogtheta: ', num2str(abs(flogtheta(end) - flogtheta(end-1)))])); 
    disp(' ');
    
    [hyp, flogthetatmp,i] = feval(minf,hyp, @gp, iters, inf, mean, cov, lik, input, target);
    
    if isempty(flogthetatmp) % no improvement: at minimum
        disp('oops');
        break;
    end
    flogtheta = [flogtheta flogthetatmp(end)];
end

