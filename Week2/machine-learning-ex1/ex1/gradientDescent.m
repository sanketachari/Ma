function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	cost = zeros(2,1);
	for i = 1:m
        cost(1,1) = cost(1,1) + (theta(1) * X(i,1) + theta(2) * X(i,2) - y(i)) * X(i,1);
        cost(2,1) = cost(2,1) + (theta(1) * X(i,1) + theta(2) * X(i,2) - y(i)) * X(i,2);
    end
    
    theta(1,1) = theta(1,1) - alpha*cost(1)/m;
    theta(2,1) = theta(2,1) - alpha*cost(2)/m;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	
% 	fprintf('Cost J_history(%d): %f', iter, J_history(iter));    

end

end
