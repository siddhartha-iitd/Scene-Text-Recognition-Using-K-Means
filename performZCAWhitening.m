function featuresMatrix_ZCAwhite = performZCAWhitening(featuresMatrix)
%ZCAWhitening
% Refer to http://ufldl.stanford.edu/wiki/index.php/Whitening
% http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening
% http://eric-yuan.me/ufldl-exercise-pca-image/

%CoVariance Matrix
coVarMatrix = featuresMatrix * featuresMatrix' / size(featuresMatrix, 2);

%Compute the eigenvectors; matrix U will contain the eigenvectors of coVarMatrix; 
%Diagonal entries of the matrix S will contain the corresponding eigenvalues; V=U'; 
[U,S,V] = svd(coVarMatrix);
epsilon = 10^-5;
featuresMatrix_ZCAwhite = U * diag(1./sqrt(diag(S) + epsilon)) * U' * featuresMatrix;
end