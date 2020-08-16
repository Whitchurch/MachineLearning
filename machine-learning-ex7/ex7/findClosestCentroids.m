function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
temp = size(centroids,1);
% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);
tempidx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% fprintf("%f\n",X(1,:));
% fprintf("%f\n",centroids);
% fprintf("%f\n",norm(X(1,:)-centroids(1,:),2)^2);
% fprintf("%f\n",norm(X(1,:)-centroids(2,:),2)^2);
% fprintf("%f\n",norm(X(1,:)-centroids(3,:),2)^2);

% fprintf("%f\n",norm(X(2,:)-centroids(1,:),2)^2);
% fprintf("%f\n",norm(X(2,:)-centroids(2,:),2)^2);
% fprintf("%f\n",norm(X(2,:)-centroids(3,:),2)^2);

% fprintf("%f\n",norm(X(3,:)-centroids(1,:),2)^2);
% fprintf("%f\n",norm(X(3,:)-centroids(2,:),2)^2);
% fprintf("%f\n",norm(X(3,:)-centroids(3,:),2)^2);
for iter1 = 1:size(X,1)
for iter2 = 1:K
 temp(iter2) = (X(iter1,:)-centroids(iter2,:))*(X(iter1,:)-centroids(iter2,:))';
 
 % fprintf("value %f",val);
 % fprintf("index %f",idxval);
 % fprintf("%f\n",temp(iter2));
end
[val,idxval] = min(temp');
tempidx(iter1) = idxval;
end
idx = tempidx;

% Note: You can use a for-loop over the examples to compute this.
%







% =============================================================

end

