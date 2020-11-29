function labelContext = getLabelContext( ...
  mode, numClass, upperBound, matrixFlag)

if ~ismember(mode, {'L', 'Q', 'TL', 'TQ', 'Cauchy', 'fly', 'Huber'})
  error('Unsupported label context type') ;
end

if isempty(upperBound) && ~ismember(mode, {'L', 'Q', 'fly'})
  error('Bound is empty, while label context mode is not fly') ;
end

if matrixFlag
  iMax = numClass ;
  jMax = numClass ;
else
  iMax = 1 ;
  jMax = numClass ;
end

labelContext = zeros(iMax, jMax, 'single') ;

if strcmp(mode, 'L')
  for i = 1 : iMax
    for j = 1 : jMax
      labelContext(i, j) = single(abs(i - j)) ;
    end
  end
elseif strcmp(mode, 'Q')
  for i = 1 : iMax
    for j = 1 : jMax
      labelContext(i, j) = single((abs(i - j)^2)) ;
    end
  end
elseif strcmp(mode, 'TL')
  for i = 1 : iMax
    for j = 1 : jMax
      labelContext(i, j) = single(min(abs(i - j), upperBound)) ;
    end
  end
elseif strcmp(mode, 'TQ')
  for i = 1 : iMax
    for j = 1 : jMax
      tmp = min(abs(i - j), upperBound) ;
      labelContext(i, j) = single(tmp^2) ;
    end
  end
elseif strcmp(mode, 'Cauchy')
  for i = 1 : iMax
    for j = 1 : jMax
      labelContext(i, j) = ...
        0.5 * single(upperBound)^2 * log(single(1 + ((i-j) / upperBound)^2)) ;
    end
  end
elseif strcmp(mode, 'Huber')
  for i = 1 : iMax
    for j = 1 : jMax
      absDiff = abs(i - j) ;
      if absDiff <= upperBound
        labelContext(i, j) = absDiff^2 ;
      else
        labelContext(i, j) = 2 * upperBound * absDiff - upperBound^2 ;
      end
    end
  end
elseif strcmp(mode, 'fly')
  randValue = sort(randi(100, 1, numClass, 'single')) ;
  randValue(1, 1) = 0 ;
  for i = 1 : iMax
    for j = 1 : jMax
      labelContext(i, j) = randValue(abs(i - j) + 1) ;
    end
  end
end

labelContext = int32(labelContext) ;
