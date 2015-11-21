load('centroids.mat');
stride = 1;
patchSize=8; % 8X8 patch size
windowSize=[32 32];
K=size(centroids,1);

% go to + training images dir, load the images, resize them to 32x32.
% Get 4K feature vector for each image, put label of image in the 4K+1 column.

tPosImgSet = getAllFiles('train_images/Pos');
NPos=numel(tPosImgSet);
for i=1:NPos
  tPosImgSet{i}= rgb2gray(imresize(imread(tPosImgSet{i}),windowSize));
end;

tPosImgFeatures = zeros(NPos, 4*K + 1);
tPosImgFeatures(:, 1:4*K) = getImgFeatureVector(tPosImgSet, centroids, patchSize, stride);
tPosImgFeatures(:, 4*K +1) = 1;

% go to - training images dir, load the images, resize them to 32x32.
% Get 4K feature vector for each image, put label of image in the 4K+1 column.
tNegImgSet = getAllFiles('train_images/Neg');
NNeg=numel(tNegImgSet);
for i=1:NNeg
  tNegImgSet{i}= rgb2gray(imresize(imread(tNegImgSet{i}),windowSize));
end;

tNegImgFeatures = zeros(NNeg, 4*K + 1);
tNegImgFeatures(:, 1:4*K) = getImgFeatureVector(tNegImgSet, centroids, patchSize, stride);
tNegImgFeatures(:, 4*K +1) = 0;

tImgFeatures = [tPosImgFeatures; tNegImgFeatures];
% Randomly Shuffle the Features Matrix.
trainingSetFeatures = tImgFeatures(randperm(size(tImgFeatures,1)),:);
% Now, pass these labeled 4K feature matrix to Linear Classifier SVM
