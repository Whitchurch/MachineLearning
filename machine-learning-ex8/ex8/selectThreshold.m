function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

predictions = zeros(size(yval,1),1);
tp = 0;
fp = 0;
fn = 0;

prec = 0;
rec = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

     
	 predictions = (pval < epsilon);
	 tp = sum((predictions==1)&(yval ==1));
	 fp = sum((predictions==1)&(yval ==0));
	 fn = sum((predictions==0)&(yval ==1));
	 
	 % tp = find(predictions == yval); %gives location of all true positives
	 
	 % for iter1 = 1:size(yval,1)
	 	   % if(predictions(iter1) ==1)
		   % if(yval(iter1) == 1)
		   % tp = tp+1;
		   % end
	   % end
	 % end
	 
	 % for iter1 = 1:size(yval,1)
	   % if(predictions(iter1) ==1)
		   % if(yval(iter1) == 0)
		   % fp = fp+1;
		   % end
	   % end
	 % end
	 
	 
	 	 % for iter1 = 1:size(yval,1)
	   % if(predictions(iter1) ==0)
		   % if(yval(iter1) == 1)
		   % fn = fn+1;
		   % end
	   % end
	 % end

prec = (tp)/(tp+fp);
rec = (tp)/(tp+fn);

F1 = (2*prec*rec)/(prec+rec);

	 
	% pos = find(predictions==1);
    % fprintf("%f\n",size(pos));
	
	% fprintf("\n stepsize: %f",stepsize);
	% fprintf("\n epsilon: %f",epsilon);
	% fprintf("\n min-pval: %f",min(pval));
	% fprintf("\n max-pval: %f",max(pval));
% fprintf("pval size %f\n",size(pval));
% fprintf("yval size %f",size(yval));










    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
