function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);
meansigmavector = ones(1,m);
tempsigma2 = zeros(n,1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


% fprintf("%f\t%f",X(1,1),X(1,2));
% fprintf("\n Row: %f",m);
% fprintf("\n Column: %f",n);
% fprintf("\n X dimensions: %f",size(X));
% fprintf("\n meansigma dimensions: %f",size(meansigmavector));

result = meansigmavector*X;
fprintf("\n result: %f",size(result'));
fprintf("\n mu: %f",size(mu));

mu = (result')./m;

% fprintf("mu size: %f",size(mu));

for i = 1:n
 for j = 1:m
  tempsigma2(i) = tempsigma2(i)+(X(j,i)-mu(i))^2;
 end
end

sigma2 = (tempsigma2)./m;
% fprintf("tempsigma2 size: %f",size(tempsigma2));
% fprintf("tempsigma2_1 %f",tempsigma2(1));
% fprintf("tempsigma2_2 %f",tempsigma2(2));
% =============================================================


end
