function featureMap = getFeatureMapping(centroids, img)
[m n] = size(img);
K=size(centroids,1);
z=zeros(1,K);
featureMap=zeros(1,K);
zero = zeros(1,K);
%For each centroid j, compute the distance of img(:)` from that centroid. 
%Store all these distances in z
for j=1:K
  z(1,j)=norm((img(:)')-centroids(j,:));
end
%Compute the mean distance of img from all centroids
meanVal=mean(z);
%Compute the feature mapping for img
featureMap=max(zero, -z .+ meanVal);
end