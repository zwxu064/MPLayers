%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function errorPixel = calErrorPixel(seg, gt)
seg = double(seg);
gt = double(gt);
nPixels = size(seg, 1) * size(seg, 2);

excludeArea = (gt == 0);
excludeNum = sum(sum(excludeArea));

errorPixel = sum(sum(abs(seg - gt) .* (1 - excludeArea)));
errorPixel = errorPixel / (nPixels - excludeNum);