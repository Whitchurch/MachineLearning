function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% fprintf("X %f\n",size(X));
% fprintf("y %f\n",size(y));
% fprintf("Xval %f\n",size(Xval));
% fprintf("yval %f\n",size(yval));
predictionBestValue = 1000;
x1 = [1 2 1]; x2 = [0 4 -1];

cValues = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmaValues = [0.01 0.03 0.1 0.3 1 3 10 30];

cValues = cValues(:);
sigmaValues = sigmaValues(:);

CValueFinal = 0;
sigmaValueFinal = 0;

% fprintf("cValues %f\n",cValues(1));
% fprintf("sigmaValues %f\n",sigmaValues(1));

for i = 1:size(cValues,1)
CValueUsed = cValues(i);
	for j = 1:size(sigmaValues,1)
	sigmaValueUsed = sigmaValues(j);
	model= svmTrain(X, y, CValueUsed, @(x1, x2) gaussianKernel(x1, x2, sigmaValueUsed));
	predictions = svmPredict(model, Xval);
	predictionError =  mean(double(predictions ~= yval));
	
	if(predictionError < predictionBestValue)
		{
		fprintf("predictionError %f\n",predictionError);
		fprintf("CValueUsed %f\n",CValueUsed);
		fprintf("sigmaValueUsed %f\n",sigmaValueUsed);
		predictionBestValue = predictionError;
		 CValueFinal = CValueUsed;
		 sigmaValueFinal = sigmaValueUsed;
		}
		end 
	end
end

C = CValueFinal;
sigma = sigmaValueFinal;
% =========================================================================

end
