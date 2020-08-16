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

z = [];
z = theta'*X';

fprintf('%f\n',size(theta'(1,(2:size(theta',2)))));
thetaJ = theta'(1,(2:size(theta',2)));
J = (1/m)*(-y'*log(1./(1.+exp(-z)))'-(1.-y)'*log(1.-(1./(1.+exp(-z))))')+(0.5.*(lambda/m))*(thetaJ*thetaJ');
%J = (1/m)*(-y'*log(1./(1.+exp(-z)))'-(1.-y)'*log(1.-(1./(1.+exp(-z))))')+(0.5.*(lambda/m))*(theta'*theta);

%J = ((1/m)*(-y'*log(1./(1.+exp(-z)))'-(1.-y)'*log(1.-(1./(1.+exp(-z))))'))+(0.5)*(thetaJ*thetaJ');

%fprintf('%f\n',size(J));

 % for n = 2:size(theta)
% J = (1/m)*(-y'*log(1./(1.+exp(-z)))'-(1.-y)'*log(1.-(1./(1.+exp(-z))))')+(0.5.*(lambda/m))*(theta'(n)*theta(n));
 % end
 

z1 = [];
z1 = theta'*X';
alpha = -1
grad(1) = grad(1)-((alpha/m)*((1./(1.+exp(-z1))-y')*X(:,1)))';

 for n = 2:size(theta)
 grad(n) = grad(n)-((alpha/m)*((1./(1.+exp(-z1))-y')*X(:,n)))'+(lambda/m).*(theta(n));
 end
%fprintf('%f\n',size(theta));
%fprintf('%f\n',size(X'(1,:)));
%$thetareg = (lambda/m).*(theta)
%fprintf('%f\n',size(X(1,:));
%fprintf('%f\n',size(grad));
%fprintf('%f\n',grad);
% =============================================================

end
