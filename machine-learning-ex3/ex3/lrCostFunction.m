function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%

% fprintf('\n number of training samples %f',m);
% fprintf('\n dimension of the y matrix');
% fprintf('%f\n',size(y));

% fprintf('\n dimension of the x matrix');
% fprintf('%f\n',size(X));

% fprintf('\n dimension of the theta matrix');
% fprintf('%f\n',size(theta));



z = [];
z = theta'*X';

%fprintf('%f\n',size(theta'(1,(2:size(theta',2)))));
thetaJ = theta'(1,(2:size(theta',2)));
J = (1/m)*(-y'*log(1./(1.+exp(-z)))'-(1.-y)'*log(1.-(1./(1.+exp(-z))))')+(0.5.*(lambda/m))*(thetaJ*thetaJ');

alpha = -1;
z1 = [];
z1 = theta'*X';
grad(1) = grad(1)-((alpha/m)*((1./(1.+exp(-z1))-y')*X(:,1)))';

 % for n = 2:size(theta)
 % grad(n) = grad(n)-((alpha/m)*((1./(1.+exp(-z1))-y')*X(:,n)))'+(lambda/m).*(theta(n));
 % end
 
  
 
 gradT = zeros(size(theta)-1,1);
 thetaJ = theta'(1,(2:size(theta',2)));
 
 XT = X(:,(2:size(theta',2)));
  gradT = gradT-((alpha/m)*((1./(1.+exp(-z1))-y')* XT))'+(lambda/m).*(thetaJ');
  
 grad(2:size(theta',2),1) = gradT;
  
    % fprintf('\n X \n');
  % fprintf('%f\n', size(X));
  
      % fprintf('\n XT \n');
  % fprintf('%f\n', size(XT));
  
  % fprintf('\n Y \n');
  % fprintf('%f\n', size(y));
  
   % fprintf('\n grad \n');
  % fprintf('%f\n', size(grad));
  
  % fprintf('\n gradT \n');
  % fprintf('%f\n', size(gradT));
  
    % fprintf('\n gradReg \n');
  % fprintf('%f\n', size(gradT));
  % fprintf('%f\n', gradT);
  
    % fprintf('\n theta \n');
  % fprintf('%f\n', size(theta));
  
      % fprintf('\n thetaJ \n');
  % fprintf('%f\n', size(thetaJ));
  % fprintf('%f\n', length(2:size(theta)));

%grad(2:size(theta)) = grad(2:size(theta))-((alpha/m)*((1./(1.+exp(-z1))-y')*X(:,2:size(theta))))'+(lambda/m).*(theta(2:size(theta)));

% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

grad = grad(:);

end
