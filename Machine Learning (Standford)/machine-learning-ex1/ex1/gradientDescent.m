function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);
J_history = zeros(num_iters, 1);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

for iter = 1:num_iters
    delta = zeros(n,1);
    for i = 1:n
        for j = 1:m
            delta(i) = delta(i) + (theta'*X(j,:)'-y(j))*X(j,i);
        end
    end
    delta = delta/m;
    theta = theta-alpha*delta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end
end