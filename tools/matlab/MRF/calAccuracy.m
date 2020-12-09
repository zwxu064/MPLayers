%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function accuracy = calAccuracy(seg, gt, accThresh)
seg = double(seg);
gt = double(gt);
nPixels = size(seg, 1) * size(seg, 2);

excludeArea = (gt == 0);
excludeNum = sum(sum(excludeArea));

accurateNum = sum(sum(abs(seg - gt) .* (1 - excludeArea) <= accThresh));
accuracy = (accurateNum - excludeNum) / (nPixels - excludeNum);