function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

%number of training samples
% m = length(y); % number of training examples

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
% for iter = 1:m

% if(y(iter) == 0)
% plot(X(iter,1),X(iter,2),'ko','MarkerSize',10);
% else
% plot(X(iter,1),X(iter,2),'k+','MarkerSize',10);
% end;
% end

pos = find(y==1);
neg = find(y==0);

plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2,'MarkerSize', 7); 

plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);

% =========================================================================



hold off;

end
