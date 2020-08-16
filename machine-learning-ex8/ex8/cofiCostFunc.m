function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
TotalCost = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));
Theta_gradTemp = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% fprintf("X: %f\n",size(X));
% fprintf("Theta: %f\n",size(Theta));
% fprintf("Y: %f\n",size(Y));
% fprintf("R: %f\n",size(R));

% disp(X);
 % disp(Theta);
% % disp(Y);

 disp(R);

for j = 1:size(R,2)
 % disp(Theta(j,:));
% disp(X.*R(:,j));
% disp((X.*R(:,j))*(Theta(j,:))');
% disp(Y(:,j));
% disp(TotalCost = TotalCost+sum((((X.*R(:,j))*(Theta(j,:))')-Y(:,j)).^2));
% disp((0.5).*(TotalCost));
% fprintf("size Xreduced: %f \n",size((((X.*R(:,j))*(Theta(j,:))')-Y(:,j)).^2));
% fprintf("size Y: %f \n",size(Y(:,j)));

% % TotalCost = TotalCost+ (((X.*R(:,j))*(Theta(j,:))')-Y(:,j))^2;
TotalCost = TotalCost+sum((((X.*R(:,j))*(Theta(j,:))')-Y(:,j)).^2);


end

% disp(X_grad);
% disp(Theta_grad);
% disp(Y);
 % fprintf("size X_grad: %f \n",size(X_grad));
 % fprintf("size Theta_grad: %f \n",size(Theta_grad));
J = ((0.5).*(TotalCost))+((lambda/2).*(sum(sum(Theta.^2))))+((lambda/2).*(sum(sum(X.^2))));


for j = 1:size(R,2)
% % fprintf("\n size: %f",size((((X.*R(:,j))*(Theta(j,:))')-Y(:,j))));
% % disp((((X.*R(:,j))*(Theta(j,:))')-Y(:,j)));

% fprintf("\n size X: %f",size(X.*R(:,j)));
% disp(X.*R(:,j));

% fprintf("\n size: %f",size(((((X.*R(:,j))*(Theta(j,:))')-Y(:,j))'*(X.*R(:,j)))));
% disp(((((X.*R(:,j))*(Theta(j,:))')-Y(:,j))'*(X.*R(:,j))));
Theta_grad(j,:) = (((((X.*R(:,j))*(Theta(j,:))')-Y(:,j))'*(X.*R(:,j))))+(lambda.*Theta(j,:));
end

for j = 1:size(R,1)
% fprintf("\n Y size: %f",size(Y(j,:)))
% fprintf("\n Theta size: %f",size(Theta.*R(j,:)'));
% disp(Theta.*R(j,:)');
% disp(X(j,:));
% disp(Y(j,:));

% disp((X(j,:))*(Theta.*R(j,:)')');

% disp((((X(j,:))*(Theta.*R(j,:)')')-(Y(j,:)))*((Theta.*R(j,:)')')');
% fprintf("\n size: %f",size((((X(j,:))*(Theta.*R(j,:)')')-(Y(j,:)))*((Theta.*R(j,:)')')'));
% fprintf("\n size: %f",size((((X.*R(:,j))*(Theta(j,:))')-Y(:,j))));
% disp((((X.*R(:,j))*(Theta(j,:))')-Y(:,j)));
% disp(Y(:,j));
% fprintf("\n size X: %f",size(Theta(j,:)));
% disp(Theta(j,:));
% Theta_gradTemp = zeros(size(Theta));
% Theta_gradTemp(j,:) = Theta(j,:);
% disp(Theta_gradTemp);
% fprintf("\n size Theta_gradTemp: %f",size(Theta_gradTemp));
% TotalValue = ((((X.*R(:,j))*(Theta(j,:))')-Y(:,j))*(Theta(j,:)));
% fprintf("\n size: %f",size(TotalValue));
% disp(TotalValue);
X_grad(j,:) = ((((X(j,:))*(Theta.*R(j,:)')')-(Y(j,:)))*((Theta.*R(j,:)')')')+(lambda.*X(j,:));
end
fprintf("size X_grad: %f",size(X_grad));

% X_grad = X_grad +((((X.*R(:,j))*(Theta(j,:))')-Y(:,j))*(Theta(j,:)));









% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
