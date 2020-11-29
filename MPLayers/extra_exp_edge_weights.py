import torch
import torch.nn.functional as F
import os
import time
import scipy.io as scio
import numpy as np
import argparse

from MP_module import MPModule
from test_compute_terms import MRFParams, compute_terms_py


os.environ["CUDA_VISIBLE_DEVICES"]="3"
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
enable_cuda = torch.cuda.is_available()
device = torch.device('cuda' if enable_cuda else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
enable_backward = False


def get_steps(dir):
  if dir == 0:
    h_step, w_step = 0, 1
  elif dir == 1:
    h_step, w_step = 0, -1
  elif dir == 2:
    h_step, w_step = 1, 0
  elif dir ==3:
    h_step, w_step = -1, 0
  elif dir == 4:
    h_step, w_step = 1, 1
  elif dir == 5:
    h_step, w_step = -1, -1
  elif dir == 6:
    h_step, w_step = 1, -1
  elif dir == 7:
    h_step, w_step = -1, 1
  elif dir == 8:
    h_step, w_step = 1, 2
  elif dir == 9:
    h_step, w_step = -1, -2
  elif dir == 10:
    h_step, w_step = 2, 1
  elif dir == 11:
    h_step, w_step = -2, -1
  elif dir == 12:
    h_step, w_step = 2, -1
  elif dir == 13:
    h_step, w_step = -2, 1
  elif dir == 14:
    h_step, w_step = 1, -2
  elif dir == 15:
    h_step, w_step = -1, 2
  else:
    print('Warning, edge weights support only 16 directions by far')
    h_step, w_step = 0, 0

  return h_step, w_step


# Copy from my own dataloader_stereo.py in MPCNN
def multi_edge_weights(img, n_dirs, threshold, penalty, scale_list=None,
                       sigma=10, enable_kernel_cue=False):
  img_group_scale = []
  channel, height, width = img.size()

  # ==== Downsample
  for scale in scale_list:
    h_size, w_size = int(height * scale), int(width * scale)

    if scale == 1:
      img_group_scale.append(img)
    else:
      w_grid = torch.linspace(-1, 1, w_size).repeat(h_size, 1)
      h_grid = torch.linspace(-1, 1, h_size).view(-1, 1).repeat(1, w_size)
      grid = torch.cat((w_grid.unsqueeze(2), h_grid.unsqueeze(2)), 2)
      img_scale = F.grid_sample(img.unsqueeze(0), grid.unsqueeze(0))
      img_group_scale.append(img_scale.squeeze(0))

  # ==== Dirs
  # img_scale:(batch,d,h,w)
  # img_scale_shift:(batch,n_dir,d,h,w)
  img_group_scale_shift = []

  for img_scale in img_group_scale:
    height, width = img_scale.size()[-2:]
    img_scale_shift = img_scale.new_zeros((n_dirs, height, width))

    for dir in range(n_dirs):
      h_step, w_step = get_steps(dir)

      if h_step > 0:
        src_h_start, src_h_stop = 0, height - h_step
        tar_h_start, tar_h_stop = h_step, height
      else:
        src_h_start, src_h_stop = -h_step, height
        tar_h_start, tar_h_stop = 0, height + h_step

      if w_step > 0:
        src_w_start, src_w_stop = 0, width - w_step
        tar_w_start, tar_w_stop = w_step, width
      else:
        src_w_start, src_w_stop = -w_step, width
        tar_w_start, tar_w_stop = 0, width + w_step

      img_patch_shifted = img_scale[:, src_h_start:src_h_stop, src_w_start:src_w_stop]
      img_patch = img_scale[:, tar_h_start:tar_h_stop, tar_w_start:tar_w_stop]

      if enable_kernel_cue:
        grad = torch.exp(torch.mean(torch.pow(img_patch - img_patch_shifted, 2), dim=0) / (-2 * sigma ** 2))
      else:
        grad = torch.mean(torch.pow(img_patch - img_patch_shifted, 2), dim=0)
        under_threshold = grad < threshold
        above_threshold = grad >= threshold
        grad[under_threshold] = penalty
        grad[above_threshold] = 1

      img_scale_shift[dir, tar_h_start:tar_h_stop, tar_w_start:tar_w_stop] = grad

    img_group_scale_shift.append(img_scale_shift)

  return img_group_scale_shift


if __name__ == '__main__':
  # img_names = ["tsukuba", "teddy", "venus", "cones", "map", "000002_11",
  #              "000041_10", "000119_10", "delivery_area_1l", "facade_1s"]
  # p_funcs = ["TL", "TL", "TQ", "TL", "TL", "TL", "TL", "TL", "TL", "TL"]
  # n_disps = [16, 60, 20, 55, 29, 96, 96, 96, 32, 32]
  # truncs = [2, 1, 7, 8, 6, 95, 95, 95, 31, 31]
  # p_weights = [20, 10, 50, 10, 4, 10, 10, 10, 10, 10]
  # # in Middlebury Eval2, only Tsukuba and Teddy have cue, others have no edge weights
  # cue_thresholds = [8, 10, 10, 10, 10, 10, 10, 10, 10, 10]
  # cue_penalties = [2, 3, 2, 2, 2, 2, 2, 2, 2, 2]
  # modes = ["ISGMR", "TRWP"]
  # n_dirs = [4, 8, 16]
  # n_iter = 1
  # enable_run_all = True
  # enable_edge_weights = True
  # enable_overwrite = False
  # enable_kernel_cue = False

  # in Middlebury Eval2, only Tsukuba and Teddy have cue, others have no edge weights
  enable_edge_weights = False
  enable_overwrite = False

  parser = argparse.ArgumentParser(description='Funny Witch')
  parser.add_argument('--server', type=str, default='data61', choices={'data61', '039614'})
  parser.add_argument('--img_name', type=str, default='tsukuba')
  parser.add_argument('--mode', type=str, default='TRWP')
  parser.add_argument('--n_dirs', type=int, default=4)
  parser.add_argument('--n_iters', type=int, default=50)
  parser.add_argument('--context', type=str, default='TL')
  parser.add_argument('--enable_min_a_dir', action='store_true', default=False)
  parser.add_argument('--enable_saving_label', action='store_true', default=False)
  parser.add_argument('--truncated', type=int, default=2)
  parser.add_argument('--n_disps', type=int, default=16)
  parser.add_argument('--p_weight', type=int, default=20)
  parser.add_argument('--cue_threshold', type=int, default=8)
  parser.add_argument('--cue_penalty', type=int, default=2)
  parser.add_argument('--edge_mode', type=str, default='', choices={'', 'kernel', 'threshold'})
  args = parser.parse_args()
  args.rho = 1 if (args.mode in ['ISGMR', 'SGM']) else 0.5

  args.n_classes = args.n_disps
  args.mpnet_n_dirs = args.n_dirs
  args.mpnet_mrf_mode = args.mode
  args.mpnet_max_iter = args.n_iters
  args.mpnet_smoothness_mode = args.context
  args.mpnet_smoothness_trunct_loc = args.truncated
  args.mpnet_smoothness_trunct_value = args.truncated
  args.mpnet_term_weight = args.p_weight
  args.enable_cuda = False if args.mode == 'SGM' else enable_cuda
  args.mpnet_smoothness_train = None

  # Compute terms
  if args.img_name[0:3] == '000':
    left_img_path = '../data/KITTI2015/image_2/{}.png'.format(args.img_name)
    right_img_path = '../data/KITTI2015/image_3/{}.png'.format(args.img_name)
  elif args.img_name.split('_')[-1][-2:] in ['1l', '2l', '3l', '1s', '2s', '3s']:
    left_img_path = '../data/ETH3D/training/{}/im0.png'.format(args.img_name)
    right_img_path = '../data/ETH3D/training/{}/im1.png'.format(args.img_name)
  else:
    postfix = 'pgm' if args.img_name == 'map' else 'ppm'
    left_img_path = '../data/{}/Middlebury/middlebury/{}/imL.{}'.format(args.server, args.img_name, postfix)
    right_img_path = '../data/{}/Middlebury/middlebury/{}/imR.{}'.format(args.server, args.img_name, postfix)

  # Create directories
  if enable_edge_weights:
    mode_str = '{}'.format(args.mode)

    if args.edge_mode == 'kernel':
      save_dir = '../exp/WithEdgeWeights-kernel'
    else:
      save_dir = '../exp/WithEdgeWeights-threshold'
  else:
    mode_str = '{}'.format(args.mode)
    save_dir = '../exp/WithOutEdgeWeights'

  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  save_dir = os.path.join(save_dir, args.img_name)

  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  file_path = os.path.join(save_dir,
                           '{}_{}_iter_{}_{}_trunc_{}_dir_{}_rho_{}.mat' \
                           .format(args.img_name, mode_str, args.n_iters,
                                   args.context, args.truncated, args.n_dirs, args.rho))

  if os.path.exists(file_path) and (not enable_overwrite):
    print('Skip {}.'.format(file_path))

  # Get params
  param = MRFParams(left_img_path, right_img_path, args.context, args.n_disps, args.cue_threshold,
                    args.cue_penalty, args.truncated)
  data_cost, RGB, smoothness_context, param = compute_terms_py(param)
  smoothness_context *= args.p_weight

  # ==== Inference
  h, w, = param.height, param.width

  # ==== Auto
  smoothness_context = smoothness_context.view(1, args.n_disps, args.n_disps).repeat(args.n_dirs, 1, 1)
  label_context = smoothness_context.contiguous().cuda()
  unary = data_cost.permute(2, 0, 1).contiguous()
  unary = unary.view(1, 1, args.n_disps, h, w).cuda()
  unary.requires_grad = False

  time_start = time.time()

  mp_module = MPModule(args,
                       enable_create_label_context=True,
                       enable_saving_label=False,
                       enable_min_a_dir=False)

  if enable_edge_weights:
    RGB = RGB.permute(2, 0, 1)
    edge_weight_group = multi_edge_weights(RGB, args.n_dirs, args.cue_threshold, args.cue_penalty,
                                           scale_list=[1], enable_kernel_cue=(args.edge_mode == 'kernel'))
    edge_weights = edge_weight_group[0].unsqueeze(0).cuda()  # (batch,n_dir,h,w)
  else:
    edge_weights = None

  mp_module.eval()

  if args.mode == 'SGM':
    unary = unary.cpu()
    edge_weights = edge_weights.cpu() if edge_weights is not None else edge_weights

  results = mp_module(unary, edge_weights=edge_weights)
  final_cost = results[0]

  final_seg = final_cost.squeeze().permute(1, 2, 0).detach().cpu().numpy()
  final_seg = np.expand_dims(np.argmin(final_seg, 2).astype(np.int8), 2)
  print('image: {}, mode: {}, dir: {}, time: {:.4f}' \
        .format(args.img_name, args.mode, args.n_dirs, time.time() - time_start))

  if enable_edge_weights:
    scio.savemat(file_path,
                 {'n_iter': args.n_iters,
                  'n_dir': args.n_dirs,
                  'rho': args.rho,
                  'p_func': args.context,
                  'n_disp': args.n_disps,
                  'p_weight': args.p_weight,
                  'trunct': args.truncated,
                  'min_a_dir': False,
                  'unary': unary.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                  'label_context': smoothness_context.permute(1, 2, 0).detach().cpu().numpy(),
                  'edge_weights': edge_weights.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                  'seg_all': final_seg})
  else:
    scio.savemat(file_path,
                 {'n_iter': args.n_iters,
                  'n_dir': args.n_dirs,
                  'rho': args.rho,
                  'p_func': args.context,
                  'n_disp': args.n_disps,
                  'p_weight': args.p_weight,
                  'trunct': args.truncated,
                  'min_a_dir': False,
                  'unary': unary.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                  'label_context': smoothness_context.permute(1, 2, 0).detach().cpu().numpy(),
                  'seg_all': final_seg})

  torch.cuda.empty_cache()