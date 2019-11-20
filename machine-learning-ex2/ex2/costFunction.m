function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%


%theta is  n+1 X 1 so 3 X 1
%X is m X 3, first col is all 1's
%y is m X 


h = X * theta % results in (m X 1)
h1 = sigmoid(h) %(m X 1)

error = h1 - y % (m X 1)

grad = X' * error/m


firstLog = log(h1)
fProduct = y' * firstLog

secondLog = log(1-h1)
secondProduct = (1-y)' * secondLog
J = (fProduct + secondProduct)/(-m)





% =============================================================

end
