function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%               
fprintf("Z dimensions: %f\n",size(Z));
fprintf("U dimensions: %f\n",size(U));
fprintf("K dimensions: %f\n",size(K));
fprintf("X_rec dimensions: %f\n",size(X_rec));

U_reduce = U(:, 1:K);
fprintf("U_reduce dimensions: %f\n",size(U_reduce));

for i = 1: size(Z, 1)
  X_rec(i, :) = (U_reduce*Z(i, :)');
end
% =============================================================

end
