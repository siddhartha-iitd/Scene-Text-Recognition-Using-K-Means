function grayImagePatches = extractPatchesFromImage(imageList, patchPerImage, L)
grayImagePatches= zeros(patchPerImage*numel(imageList),L,L);
for i=1:numel(imageList)
  % Read Image
  img=imread(imageList{i});
  [n m o]= size(img);
  % Generate 'patchPerImage' random integers between 1 & n-L+1 (rows); 1 & m-L+1(cols)
  randRow=randi(n-L+1,patchPerImage,1);
  randCol=randi(m-L+1,patchPerImage,1);
  % Take L consecutive rows for each random number in randRow.
  % Similarly, take L consecutive columns for each random number in randCol.
  % This set of L rows and L columns will constitute one image-patch. 
  for j=1:patchPerImage
    try
      %figure(j);
      %subplot(1,2,1), imshow(img(randRow(j)+(1:L),randCol(j)+(1:L),:));
      %subplot(1,2,2), imshow(rgb2gray(img(randRow(j)+(1:L),randCol(j)+(1:L),:)));
      grayImagePatches(j+ patchPerImage*(i-1),:,:)= rgb2gray(img(randRow(j)+(1:L),randCol(j)+(1:L),:));
    catch
      continue;
    end_try_catch
  end
end;

% Random Sampling%
grayImagePatches = grayImagePatches(randperm(size(grayImagePatches,1)),:);
end
