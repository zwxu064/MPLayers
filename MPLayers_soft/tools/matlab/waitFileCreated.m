%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function waitFileCreated(filePath)
fprintf("\n====>Wait to be created: %s.\n", filePath);
count = 0;
timeout_count = 360;  % 3mins

while 1
  if ~exist(filePath, 'file')
    if count < timeout_count
      pause(0.5);
      count = count + 1;
    else
      % This actually does not work, it will wait for the end,
      % so change maxIter
      fprintf("====>Created failed, timeout: %s, count: %d.\n", ...
        filePath, count);
      break;
    end
  else
    break;
  end
end

fprintf("====>Create successfully: %s.\n", filePath);
