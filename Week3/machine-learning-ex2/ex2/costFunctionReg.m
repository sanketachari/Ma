function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

cost = 0;
h = sigmoid(X*theta);
costMatrix = y.*log(h) + (1 - y).*log(1-h);

for i=1:m
    cost = cost + costMatrix(i);          
end

n = size(X,2);
regCost = 0;

for i=2:n
    regCost = regCost + theta(i)^2;
end

J = -cost/m + lambda*regCost/(2*m);

o = size(theta,1);  

for i=1:o
    if(i==1)
        grad(i) = (X(:,i)'*(sigmoid(X*theta) - y))/m;
    else
        grad(i) = (X(:,i)'*(sigmoid(X*theta) - y))/m + lambda*theta(i)/m;
    end
end




% =============================================================

end
