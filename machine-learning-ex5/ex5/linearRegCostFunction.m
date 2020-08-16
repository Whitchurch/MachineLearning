function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
thetaJ = theta'(1,(2:size(theta',2)));
J =(1/(2*m))*(((X*theta)-y)'*((X*theta)-y))+(0.5.*(lambda/m))*(thetaJ*thetaJ');

% fprintf("size of hypothesis minus y %f\n",size(((X*theta)-y))); %12 rows 1 column;
% fprintf("size of X %f\n",size(X));

% fprintf("size of product %f\n",size(((X*theta)-y)'*X));

% fprintf("size grad %f\n",size(grad));
% fprintf("size %f\n",size((1/m)*((X*theta)-y)'*X));

 regularizationParameter_Theta1_grad = theta';
 % fprintf("size regularizationParameter_Theta1_grad %f\n",size(regularizationParameter_Theta1_grad));
 
 regularizationParameter_Theta1_grad(:,[1]) = [];
  % fprintf("size regularizationParameter_Theta1_grad %f\n",size(regularizationParameter_Theta1_grad));
  
   regMM = size(regularizationParameter_Theta1_grad,1);
 regularizationParameter_Theta1_grad = [zeros(regMM, 1) regularizationParameter_Theta1_grad];
  % fprintf("size regularizationParameter_Theta1_grad %f\n",size(regularizationParameter_Theta1_grad));
 reg1t = ((lambda/m).*(regularizationParameter_Theta1_grad));
 % fprintf("size reg1t %f\n",size(reg1t));
 grad = ((1/m)*((X*theta)-y)'*X).+(reg1t);
  % fprintf("size grad %f\n",size(grad));











% =========================================================================

grad = grad(:);

end
