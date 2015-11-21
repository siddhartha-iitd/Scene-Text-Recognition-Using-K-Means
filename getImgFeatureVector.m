function tImgFeatures = getImgFeatureVector(tImgSet, centroids, patchSize, stride)

% Initialization
N=numel(tImgSet);
K=size(centroids,1);
tImgFeatures = zeros(N, 4*K);
% Read all labeled training images
for i=1:numel(tImgSet)
  img=tImgSet{i};
  [m n] = size(img);
  % Statistical Pre-processing
  imgNormalized=featureNormalize(img);
  imgZCAWhitened=performZCAWhitening(double(img));
  rowLen=m-patchSize+1;
  colLen=n-patchSize+1;
  imgPatchFeatures=zeros(rowLen,colLen,K);
  %count=0;
  % Compute K-Feature Matrix for each patch of statistically pre-processed labeled training image img
  for j= [1:stride:rowLen]
    for k=[1:stride:colLen]
      imgPatchFeatures(j,k,:)= getFeatureMapping(centroids, imgZCAWhitened(j:j+patchSize-1, k:k+patchSize-1));
    end
  end
  rby2 = round(rowLen/2); %13
  rby2plus1 = rby2 + 1;   %14
  cby2 = round(colLen/2); %13
  cby2plus1 = cby2 + 1;   %14
  % Divide the image into 4 quadrants and create a K-Feature representation for each quadrant
  % K-Feature representation for a quadrant is given by sum of all K-Feature representations in that quadrant
  quad1 = reshape(imgPatchFeatures(1        :rby2, 1:cby2        , :),[rby2*cby2,K]);
  quad2 = reshape(imgPatchFeatures(rby2plus1:end , 1:cby2        , :),[(rowLen-rby2)*cby2,K]);
  quad3 = reshape(imgPatchFeatures(1        :rby2, cby2plus1:end , :),[rby2*(colLen-cby2),K]);
  quad4 = reshape(imgPatchFeatures(rby2plus1:end , cby2plus1:end , :),[(rowLen-rby2)*(colLen-cby2),K]);
  % ith training image has now 4K features
  tImgFeatures(i,:) = [sum(quad1) sum(quad2) sum(quad3) sum(quad4)];
end
end