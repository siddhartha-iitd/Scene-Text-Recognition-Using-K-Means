function normalizedFeaturesMatrix = featureNormalize(featuresMatrix)
[m n]=size(featuresMatrix);
%Normalization
mu = zeros(1, n);
sigma = zeros(1, n);
% Calculate mean for each feature in featuresMatrix
mu = mean(featuresMatrix);
% Calculate standard deviation for each feature in featuresMatrix
sigma = std(featuresMatrix);
% Update each feature in featuresMatrix as (featureValue - mean)/StandardDeviation
for i=1:m
    normalizedFeaturesMatrix(i,:) = (featuresMatrix(i,:) - mu)./(sigma);
end
normalizedFeaturesMatrix(isnan(normalizedFeaturesMatrix))=0;
end
