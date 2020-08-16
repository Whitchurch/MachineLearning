function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


 

 yvector = zeros(m,num_labels);
  for c = 1:m
     indexValue = y(c);
	 yvector(c,indexValue) = 1;
 end
X = [ones(m, 1) X];

for turn = 1:m
 z2 = X(turn,:)*Theta1';
 gz2 = 1./(1.+exp(-z2));

  
   X2 = [ones(1, 1) gz2];
     %fprintf('%f\n',size(X2));

  z3 = X2*Theta2';
 gz3 = 1./(1.+exp(-z3));	 %this is the final result of the hypothesis after passing through the neural network.
 

  % fprintf('%f\n',size(yvector(turn,:)));
  % fprintf('%f\n',size(log(gz3)));

  % J += (1/m)*(-yvector(turn,:)*log(gz3)')-((1.-yvector(turn,:))*log(1.-(1./(gz3)')));
  J += (1/m)*(-yvector(turn,:)*log(1./(1.+exp(-z3)))'-(1.-yvector(turn,:))*log(1.-(1./(1.+exp(-z3))))');
  
  end
  
  thetaTotal = 0;
  
  thetaJ1 = [];
  for turnTheta1 = 1:size(Theta1,1)
  Theta1ExcludingFirstElement(turnTheta1,:) = Theta1((turnTheta1),(2:size(Theta1,2)));
  thetaJ1(turnTheta1,:) = (0.5.*(lambda/m)).*(Theta1ExcludingFirstElement(turnTheta1,:)*(Theta1ExcludingFirstElement(turnTheta1,:))');
  end
  
  
  
  for turnTotalTheta1 = 1:(size(thetaJ1))
  thetaTotal += thetaJ1(turnTotalTheta1)
  end
  
 
    thetaJ2 = [];
    for turnTheta2 = 1:size(Theta2,1)
    Theta1ExcludingSecondElement(turnTheta2,:) = Theta2((turnTheta2),(2:size(Theta2,2)));
    thetaJ2(turnTheta2,:) = (0.5.*(lambda/m)).*(Theta1ExcludingSecondElement(turnTheta2,:)*(Theta1ExcludingSecondElement(turnTheta2,:))');
    end
	
	  for turnTotalTheta2 = 1:(size(thetaJ2))
  thetaTotal += thetaJ2(turnTotalTheta2)
  end
  
   % fprintf('%f\n',thetaTotal);
   
   J += thetaTotal;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

 yvector2 = zeros(m,num_labels);
  for c = 1:m
     indexValue = y(c);
	 yvector2(c,indexValue) = 1;
 end


 for turn = 1:m
  z2 = X(turn,:)*Theta1';
 gz2 = 1./(1.+exp(-z2));

 z2reverse = [ones(1, 1) z2];
  
   a2 = [ones(1, 1) gz2];
     %fprintf('%f\n',size(X2));

  z3 = a2*Theta2';
 gz3 = 1./(1.+exp(-z3));	 %this is the final result of the hypothesis after passing through the neural network.
 smallDeltaThree(turn,:) = gz3-yvector2(turn,:);
 % fprintf("%f",smallDeltaThree);
    smallDeltaTwo(turn,:) = (Theta2(:,(2:size(Theta2,2))))'*smallDeltaThree(turn,:)'.*sigmoidGradient(z2)';
   
	Theta2_grad += smallDeltaThree(turn,:)'*a2;
	Theta1_grad += smallDeltaTwo(turn,:)'*X(turn,:);
	
	
  % fprintf("delta3:  %f\n",size(smallDeltaThree));
  % fprintf("delta2:  %f\n",size(smallDeltaTwo));
  % fprintf("temp:  %f\n",size(temp));
   
   % smallDeltaTwo = smallDeltaTwo(:,(2:size(smallDeltaTwo,2)));
   % Theta2_Grad(turn,:) = smallDeltaThree(turn,:)'*a2;  
    % Theta1_Grad(turn,:) = smallDeltaTwo(turn,:)'*X(turn,:);
     
  
   
    % fprintf("------Forward propogation dimesnions-------- \n");
  % fprintf(" xvalue: %f\n",size(X(turn,:)));
  % fprintf("Theta1: %f\n",size(Theta1));
  % fprintf("Z2: %f\n",size(z2));
  % fprintf("sigmoid(z2) %f\n",size(gz2));
   % fprintf("a2: %f\n",size(a2));
  % fprintf("Theta2: %f\n",size(Theta2));
  % fprintf("Z3: %f\n",size(z3));
  % fprintf("sigmoid(z3) %f\n",size(gz3));
  % fprintf("a3 without a0:  %f\n",size(gz3));
  
  % fprintf("------Back propogation dimesnions-------- \n");
  % fprintf("delta3:  %f\n",size(smallDeltaThree(turn,:)));
  % fprintf("delta2:  %f\n",size(smallDeltaTwo(turn,:)));
  % fprintf("Theta1Grad:  %f\n",size(Theta1_grad));
  % fprintf("Theta2Grad:  %f\n",size(Theta2_grad));
  
 
  
  
 end
 
 
 regularizationParameter_Theta2_grad = zeros(size(Theta2));
 regularizationParameter_Theta1_grad = zeros(size(Theta1));
 
 regularizationParameter_Theta2_grad = Theta2;
 regularizationParameter_Theta2_grad(:,[1]) = [];
 
 regularizationParameter_Theta1_grad = Theta1;
 regularizationParameter_Theta1_grad(:,[1]) = [];
 
 regM = size(regularizationParameter_Theta2_grad,1);
 regularizationParameter_Theta2_grad = [zeros(regM, 1) regularizationParameter_Theta2_grad];
 reg2t = ((lambda/m).*(regularizationParameter_Theta2_grad));
 
 
  regMM = size(regularizationParameter_Theta1_grad,1);
 regularizationParameter_Theta1_grad = [zeros(regMM, 1) regularizationParameter_Theta1_grad];
 reg1t = ((lambda/m).*(regularizationParameter_Theta1_grad));
 
 Theta2_grad = ((1/m).*Theta2_grad)+(reg2t);  
Theta1_grad = ((1/m).*Theta1_grad)+(reg1t);  

% fprintf("reg2t:  %f\n",size(reg2t));
% fprintf("reg1t:  %f\n",size(reg1t));

% fprintf("Theta2_grad:  %f\n",size(Theta2_gradReg));
% fprintf("Theta1_grad:  %f\n",size(Theta1_gradReg));

   % fprintf("delta3:  %f\n",size(smallDeltaThree));
  % fprintf("delta2:  %f\n",size(smallDeltaTwo));
 
  
  % smallDeltaTwo = smallDeltaTwo(:,(2:size(smallDeltaTwo,2)));
  % % fprintf("%f\n",size(Theta2));
  % fprintf("%f\n",size(X));
  % fprintf("%f\n",size(Theta1));
  % fprintf("%f\n",size(Theta2));
  % fprintf("%f\n",size(a2));
  % fprintf("%f\n",size(smallDeltaTwo));
  % fprintf("%f\n",size(smallDeltaThree));
  
  
  % TotalDeltaGradient +=  


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
