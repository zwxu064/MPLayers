%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [unaryEnergy, pairwiseEnergy, energy] = calEnergy( ...
  unary, pairwise, trunct, funcName, lambda, seg)
[height, width, nLabels] = size(unary);
[segH, segW] = size(seg);

assert(isequal([height, width], [segH, segW]), ...
  sprintf('====>%d vs %d, %d vs %d', height, segH, width, segW));

unaryEnergy = 0;
pairwiseEnergy = 0;
pairwiseH = pairwise(:, :, 1);
pairwiseV = pairwise(:, :, 2);
seg = seg + 1;
seg = floor(seg);

funcMatrix = getLabelContext(funcName, nLabels, trunct, 1);
assert(max(max(seg)) <= nLabels, sprintf(['====>func: %s, max seg: ' ...
  '%d, nLabels: %d'], funcName, max(max(seg)), nLabels));

for h = 1 : height
  for w = 1 : width
    nodeLabel = seg(h, w);
    unaryEnergy = unaryEnergy + unary(h, w, nodeLabel);
  end
end

for h = 1 : height
  for w = 1 : width
    if (h <= height - 1)  % down link
      edgeLeftNodeLabel = seg(h, w);
      edgeRightNodeLabel = seg(h + 1, w);
      context = lambda * funcMatrix(edgeLeftNodeLabel, edgeRightNodeLabel);
      pairwiseEnergy = pairwiseEnergy + context * pairwiseV(h, w);
    end

    if (w <= width - 1)  % right link
      edgeLeftNodeLabel = seg(h, w);
      edgeRightNodeLabel = seg(h, w + 1);
      context = lambda * funcMatrix(edgeLeftNodeLabel, edgeRightNodeLabel);
      pairwiseEnergy = pairwiseEnergy + context * pairwiseH(h, w);
    end
  end
end

energy = unaryEnergy + pairwiseEnergy;