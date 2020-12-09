clear;
close all;
clc;

marker_order = struct('SGM', 1, 'MF', 2, 'ISGMR', 3, 'TRWP', 4, 'TRWS', 5);

img = "tsukuba";
markers = {'-+','-o','-*','-.','-x','-s','-d','-^','-v','->','-<','-p','-h'};
save_dir = fullfile('../experiments', char(img));
enable_block_minAdir = false;
checkDir(save_dir);

clean_names = {};
list = [dir(sprintf('%s/*_unary*.mat', save_dir));
        dir(sprintf('%s/*_MeanField*.mat', save_dir));
        dir(sprintf('%s/*_SGM*.mat', save_dir));
        dir(sprintf('%s/*.csv', save_dir));
        dir(sprintf('%s/*_ISGMR*.mat', save_dir));
        dir(sprintf('%s/*_TRWP*.mat', save_dir))];
n_img = size(list, 1);
energy = {};

for idx = 1 : n_img
  name = list(idx).name;
  file_path = fullfile(list(idx).folder, name);
  is_csv = strcmp(name(end-3:end), '.csv');
  name = name(1:end-4);
  
  split_obj = split(name, '_');
  offset = size(split(img, '_'), 1) - 1;
  method = split_obj{2 + offset};
  n_iter = str2num(split_obj{4 + offset});
  p_func = split_obj{5 + offset};
  trunc = str2num(split_obj{7 + offset});
  n_dir = str2num(split_obj{9 + offset});
  rho = str2num(split_obj{11 + offset});
  enable_min_a_dir = false;
  
  if rho == 1
    rho_type = 'loopy';
  else
    rho_type = 'reweight';
  end
  
  clean_name = sprintf('%s_%d', method, n_dir);

  if strcmp(split_obj(end), 'minAdir')
    enable_min_a_dir = true;
  end
  
  if enable_min_a_dir
    clean_name = strcat(clean_name, '_m');
  end
  
  clean_name = replace(clean_name, 'MeanField', 'MF');
  clean_names{end+1} = clean_name;
  energy.(clean_name) = [];
  
  if is_csv
    energy.(clean_name) = SparseTRWSEnergyFile(file_path);
  else
    obj = load(file_path);
    unary_cost = obj.unary;
    pairwise_cost = obj.label_context;
    seg_all = obj.seg_all;
    n_disp = size(unary_cost, 3);

    for iter = 1 : n_iter
      seg = seg_all(:, :, iter);
      energy.(clean_name)(end+1) = ...
        CalEnergy(unary_cost, pairwise_cost(:, :, 1), seg + 1);
      if iter == 1 || iter == n_iter
        new_save_dir =fullfile(save_dir, sprintf('%d', iter));
        checkDir(new_save_dir);
        file_path = fullfile(new_save_dir, strcat(name, '.png'));
        imwrite(uint8(single(seg) ./ (n_disp - 1) .* 255), file_path, 'png');
      end
    end
  end
end

platte = hsv(5);
legend_names = {};
figure('DefaultAxesFontSize', 20);

for idx = 1 : n_img
  clean_name = clean_names{idx};
  
  if enable_block_minAdir && contains(clean_name, "_m")
    continue;
  end
  
  legend_names{end+1} = replace(clean_name, '_', '-');
  iter_energy = energy.(clean_name);
  assert(~isempty(iter_energy), 'energy empty');

  method_name = split(clean_name, '_');
  method_name = method_name{1};
  marker_idx = marker_order.(method_name);
  plot(1:5:length(iter_energy), ...
    iter_energy(1:5:end), ...
    markers{marker_idx}, ...
    'color', platte(marker_idx, :), ...
    'LineWidth', 3, ...
    'MarkerSize', 10, ...
    'MarkerFaceColor', platte(marker_idx, :));
  grid on; hold on; axis on tight;
end

xlabel('iteration', 'FontSize', 20);
ylabel('energy', 'FontSize', 20);
lgd = legend(legend_names, 'FontSize', 20);
lgd.NumColumns = 2;
% ylim([1.2 16] * 1e6);  % delivery_area_l1 all
% ylim([1.5 3] * 1e6);  % delivery_area_l1 clean
% ylim([0.3 3] * 1e6);  % tsukuba all
% ylim([1.8 3.4] * 1e6);  % teddy all
% ylim([1.8 2.1] * 1e6);  % teddy clean
% ylim([8.1 100] * 1e6);  % 000002_11 all
% ylim([8.2 17] * 1e6);  % 000002_11 clean
% lgd.NumColumns = 1;
% ylim([1.4 6.3] * 1e6);

if enable_block_minAdir
  saveas(gcf, fullfile(save_dir, sprintf('%s_curves_clean', img)), 'epsc');
else
  saveas(gcf, fullfile(save_dir, sprintf('%s_curves_all', img)), 'epsc');
end

% =========================================================================
function energy = CalEnergy(data_cost, pairwise_cost, labeling)
%CALENERGY Summary of this function goes here
%   Detailed explanation goes here
[height, width] = size(labeling);
energy = 0;

for h = 1 : height
  for w = 1 : width
    energy = energy + data_cost(h, w, labeling(h, w));
    
    if (w < width)
      xi = labeling(h, w);
      xj = labeling(h, w + 1);
      energy = energy + pairwise_cost(xi, xj);
    end
    
    if (h < height)
      xi = labeling(h, w);
      xj = labeling(h + 1, w);
      energy = energy + pairwise_cost(xi, xj);
    end
  end
end
end

% =========================================================================
function energy = SparseTRWSEnergyFile(img_path)
  data = importdata(img_path);
  energy = data(:, 1)';
end

% =========================================================================
function checkDir(dir)
  if ~exist(dir, 'dir')
    mkdir(dir);
  end
end