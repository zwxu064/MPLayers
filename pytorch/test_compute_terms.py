import torch
import time
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from imageio import imread
from lib_stereo_slim import compute_terms


class MRFParams():
  def __init__(self, left_img_path, right_img_path, context, n_disp, grad_thresh,
               grad_penalty, truncated):
    self.left_img_path = left_img_path
    self.right_img_path = right_img_path
    self.context = context
    self.grad_thresh = grad_thresh
    self.grad_penalty = grad_penalty
    self.truncated = truncated
    self.n_disp = n_disp
    self.height = 0
    self.width = 0
    self.n_channels = 0
    self.data_type = torch.float32


def compute_terms_py(param):
  data_type = param.data_type
  img_left = imread(param.left_img_path)

  if len(img_left.shape) == 2:
    height, width = img_left.shape
    n_channels = 1
  else:
    height, width, n_channels = img_left.shape

  n_nodes = height * width
  param.height, param.width, param.n_channels = height, width, n_channels
  n_disp = param.n_disp

  data_cost = torch.ones(n_nodes, n_disp, dtype=data_type)
  RGB = torch.zeros(n_nodes, n_channels, dtype=data_type)
  h_cue = torch.zeros(1, height * (width - 1), dtype=data_type)
  v_cue = torch.zeros(1, (height - 1) * width, dtype=data_type)
  smoothness_context = torch.zeros(n_disp, n_disp, dtype=data_type)

  compute_terms.compute_all_terms(param.left_img_path, param.right_img_path, param.context,
                                  param.grad_thresh, param.grad_penalty, param.truncated,
                                  height, width, param.n_channels, param.n_disp,
                                  data_cost, RGB, h_cue, v_cue, smoothness_context)
  data_cost = data_cost.view(height, width, -1)
  RGB = RGB.view(height, width, -1)

  return data_cost, RGB, smoothness_context, param


if __name__ == '__main__':
  torch.manual_seed(2019)
  torch.cuda.manual_seed_all(2019)
  img_name = 'tsukuba'

  # Compute terms
  if img_name == 'tsukuba':
    left_img_path = '../data/Middlebury/middlebury/tsukuba/imL.ppm'
    right_img_path = '../data/Middlebury/middlebury/tsukuba/imR.ppm'
    n_disp, p_weight, grad_thresh, grad_penalty = 16, 20, 8, 2
    context, truncated = 'TL', 15
  elif img_name == 'teddy':
    left_img_path = '../data/Middlebury/middlebury/teddy/imL.ppm'
    right_img_path = '../data/Middlebury/middlebury/teddy/imR.ppm'
    n_disp, p_weight, grad_thresh, grad_penalty = 60, 10, 10, 3
    context, truncated = 'TL', 59
  elif img_name[0:3] == '000':
    left_img_path = '../data/KITTI2015/image_2/{}.png'.format(img_name)
    right_img_path = '../data/KITTI2015/image_3/{}.png'.format(img_name)
    n_disp, p_weight, grad_thresh, grad_penalty = 192, 10, 10, 3
    context, truncated = 'TL', 191
  elif img_name.split('_')[-1][-1] in ['l', 's']:
    left_img_path = '../data/ETH3D/training/{}/im0.png'.format(img_name)
    right_img_path = '../data/ETH3D/training/{}/im1.png'.format(img_name)
    n_disp, p_weight, grad_thresh, grad_penalty = 64, 10, 10, 3
    context, truncated = 'TL', 63
  else:
    assert False

  param = MRFParams(left_img_path, right_img_path, context, n_disp, grad_thresh, grad_penalty, truncated)

  time_start = time.time()
  data_cost, RGB, smoothness_context, param = compute_terms_py(param)
  compute_time = (time.time() - time_start) * 1e3
  print('Data_cost: {:.4f}, compute all, time: {:.4f} ms.' \
        .format(data_cost.abs().sum(), compute_time))
  print('--------------------------------------------------------')

  height, width = param.height, param.width
  BGR = RGB.flip(2)
  predictions = data_cost.detach().numpy().argmin(2)  # Note pytorch 1.1.0 return max indices, use numpy to check
  plt.figure()
  plt.subplot(2,1,1)
  plt.imshow(BGR / 255)
  plt.subplot(2,1,2)
  plt.imshow(predictions)
  plt.show(block=False)
  plt.pause(10)
  plt.close()
