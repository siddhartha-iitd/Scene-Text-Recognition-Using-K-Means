clear all;
addpath('E:\Users\SIDDHARTHA\Downloads\MachineLearning\Text Recognition\libsvm-3.20\matlab');
pkg load image;
tic;
load('trainingSetFeatures.mat');
load('centroids.mat');
%load('svm.mat');
model = svmtrain(trainingSetFeatures(:,end) , trainingSetFeatures(:,1:end-1), '-s 0 -t 0 -c 20');
theta = model.SVs' * model.sv_coef;
b = -model.rho;
if (model.Label(1) == -1)
    theta = -theta; b = -b;
end

% Cross-Validation
iSet=getAllFiles('cv_images');
count=0;
% for each image, take all 32x32 windows and predict if it has text or not and if possible find which area of image has text
for i=1:numel(iSet)
  img= rgb2gray(imread(iSet{i}));
  [m n] = size(img);
%  if (m>1024 || n>1024)
  if (m>=64)
    img=imresize(img, [32 NaN]);
  end
%  endif
%  img=rgb2gray(img);
  [m n] = size(img);
  for j= [1:4:m-32+1]
    for k=[1:4:n-32+1]
      imgFeatures = getImgFeatureVector({img(j:j+32-1, k:k+32-1)}, centroids, 8, 1);
      hyp=imgFeatures*theta+b;
      if (hyp >=0)
        count=count+1;
        fprintf('imgName:= %s \n', iSet{i});
        fprintf('imgFeatures*theta:= %f \n', hyp);
      endif
    endfor
  endfor
endfor
toc
        
  
  



