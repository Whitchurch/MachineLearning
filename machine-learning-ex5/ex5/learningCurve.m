function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
      	  
	   % fprintf("theta %f\n",size(theta));
	   % fprintf("X %f\n",size(X));
	   % fprintf("Xval %f\n",size(Xval));
	   % fprintf("Cross validation %f\n",size(error_val));
	  
X = [ones(m, 1) X];	
Xval = [ones(size(Xval,1), 1) Xval];	  
	  for i = 1:m
          %Jtrain error, store the cost for each error inside the vector for plotting
		  n = size(X(1:i,:), 1);
		  [theta] = trainLinearReg([X(1:i,:)], y(1:i), lambda);
		  error_train(i) = (1/(2*n))*(((X(1:i,:)*theta)-y(1:i))'*((X(1:i,:)*theta)-y(1:i)));
		  error_val(i) = (1/(2*m))*(((Xval*theta)-yval)'*((Xval*theta)-yval));
		 % fprintf("Jtrain results %f\n",size((1/(2*m))*(((X(1:i,:)*theta)-y(1:i,:))'*((X(1:i,:)*theta)-y(1:i,:)))));
      end
	  
	  % fprintf("Jcross result %f\n",size((Xval*theta)-yval));

	  	  % for j = 1:n
          % %Jtrain error, store the cost for each error inside the vector for plotting
		  % error_val(j) = (1/(2*n))*(((Xval(1:j,:)*theta)-yval(1:j,:))'*((Xval(1:j,:)*theta)-yval(1:j,:)));
		 % % fprintf("Jtrain results %f\n",size((1/(2*m))*(((X(1:i,:)*theta)-y(1:i,:))'*((X(1:i,:)*theta)-y(1:i,:)))));
      % end
    




% -------------------------------------------------------------

% =========================================================================

end
