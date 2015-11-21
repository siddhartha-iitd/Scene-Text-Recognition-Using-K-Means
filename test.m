clear; clc;
% Dimension of image patch LxL.
L=8;
% No. of patches extracted per image.
patchPerImage=5;
%########################################################################################################################################%
                             % Random Patch Extraction
%########################################################################################################################################%
% getAllFiles from 'image' dir to imageList
imageList=getAllFiles('image');
%imshow(imageList{1});

% Extract 'patchPerImage' patches of dimension LxL from each image in imageList.
grayImagePatches = extractPatchesFromImage(imageList, patchPerImage, L);

%h= zeros(L,L);
%figure(1);
%for i=1:9
%  subplot(3,3,i);
%  h = imshow(reshape(grayImagePatches(i,:,:),L,L)',[]);
%  title(num2str(i));
%end

%########################################################################################################################################%
                             % Statistical Pre-processing: Normalization & ZCA-Whitening
%########################################################################################################################################%

num_rows = size(grayImagePatches,1);
featuresMatrix=zeros(num_rows,L*L);
%Convert grayImagePatches(num_rows,L,L) Matrix to 2D featuresMatrix(num_rows,L*L)
for i=1:num_rows
  featuresMatrix(i,:)=reshape(grayImagePatches(i,:,:),[1,L*L]);
end

%Normalization
featuresMatrix = featureNormalize(featuresMatrix);
%ZCAWhitening
featuresMatrix_ZCAwhite = performZCAWhitening(featuresMatrix);

%########################################################################################################################################%
                             % K-Means Clustering Feature Learning
%########################################################################################################################################%
K=800;
max_iters=10;
initial_centroids = kMeansInitCentroids(featuresMatrix_ZCAwhite, K);

% Run K-Means
[centroids, idx] = runkMeans(featuresMatrix_ZCAwhite, initial_centroids, max_iters);

save('centroids.mat','centroids');

%Spatial Pooling for dimensionality reduction

% Given a 32X32 labeled training image(I), compute a KFeatureMapping for all 8X8 patches possible in the image(625 patches)
% Split the image I into 4 equal quadrants, {0-15, 0-15}(2), {16-31, 0-15,}(1),  {16-31, 0-15}(3), {16-31, 16-31}(4)
% For each quadrant, compute the sum of all KFeatureMappings in that quadrant. This will lead to 4 KFeatureMappings for entire image, than 625 such mappings.
% Represent each image using this 4K feature vector


