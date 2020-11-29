import torch
import os
import time
import scipy.io as scio
import numpy as np
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


if __name__ == '__main__':
  img_names = ["tsukuba", "teddy", "venus", "cones", "map", "000002_11",
               "000041_10", "000119_10", "delivery_area_1l", "facade_1s"]
  p_funcs = ["TL", "TL", "TQ", "TL", "TL", "TL", "TL", "TL", "TL", "TL"]
  n_disps = [16, 60, 20, 55, 29, 96, 96, 96, 32, 32]
  truncs = [2, 1, 7, 8, 6, 95, 95, 95, 31, 31]
  p_weights = [20, 10, 50, 10, 4, 10, 10, 10, 10, 10]
  modes = ["SGM"]
  n_dirs = [4, 8, 16]
  n_iter = 50
  grad_thresh, grad_penalty = 0, 0
  enable_minus_unary = False  # Do not do this for multiple iterations due to a message normalization in my SGM implementation
  enable_norm_unary = True  # Not affact if there is already a message normalization, standard SGM has no normalization
  enable_run_all = True
  enable_overwrite = False

  assert not enable_minus_unary

  for n_dir in n_dirs:
    for mode in modes:
      rho = 1 if (mode in ['SGM']) else 0.5

      for idx in range(len(img_names)):
        img_name = img_names[idx]
        context = p_funcs[idx]
        truncated = truncs[idx]
        p_weight = p_weights[idx]
        n_disp = n_disps[idx]

        if not enable_run_all:
          if not ((img_name == 'tsukuba') and (n_dir == 4)):
            continue

        # Compute terms
        if img_name[0:3] == '000':
          left_img_path = '../data/KITTI2015/image_2/{}.png'.format(img_name)
          right_img_path = '../data/KITTI2015/image_3/{}.png'.format(img_name)
        elif img_name.split('_')[-1][-2:] in ['1l', '2l', '3l', '1s', '2s', '3s']:
          left_img_path = '../data/ETH3D/training/{}/im0.png'.format(img_name)
          right_img_path = '../data/ETH3D/training/{}/im1.png'.format(img_name)
        else:
          postfix = 'pgm' if img_name == 'map' else 'ppm'
          left_img_path = '../data/Middlebury/middlebury/{}/imL.{}'.format(img_name, postfix)
          right_img_path = '../data/Middlebury/middlebury/{}/imR.{}'.format(img_name, postfix)

        # Create directories
        name = 'ISGMNaive' if (not enable_minus_unary) else 'ISGMNaiveMinusUnary'
        save_dir = '../exp/ISGMNaive'

        if not os.path.exists(save_dir):
          os.mkdir(save_dir)

        save_dir = os.path.join(save_dir, img_name)

        if not os.path.exists(save_dir):
          os.mkdir(save_dir)

        file_path = os.path.join(save_dir,
                                 '{}_{}_iter_{}_{}_trunc_{}_dir_{}_rho_{}.mat' \
                                 .format(img_name, name, n_iter, context, truncated, n_dir, rho))

        if os.path.exists(file_path) and (not enable_overwrite):
          print('Skip {}.'.format(file_path))
          continue

        # Get params
        param = MRFParams(left_img_path, right_img_path, context, n_disp, grad_thresh,
                          grad_penalty, truncated)
        data_cost, RGB, smoothness_context, param = compute_terms_py(param)
        smoothness_context *= p_weight

        # ==== Inference
        h, w, = param.height, param.width

        # ==== Auto
        smoothness_context = smoothness_context.view(1, n_disp, n_disp).repeat(n_dir, 1, 1)
        label_context = smoothness_context.contiguous().cpu()
        unary = data_cost.permute(2, 0, 1).contiguous()
        unary = unary.view(1, 1, n_disp, h, w).cpu()
        unary.requires_grad = False

        time_start = time.time()
        mp_module = MPModule(n_dir=n_dir, n_iter=1, n_disp=n_disp, mode=mode,
                             rho=rho, label_context=label_context,
                             enable_saving_label=False,
                             enable_min_a_dir=False)
        mp_module.eval()

        final_cost, _ = mp_module(unary)

        if enable_minus_unary:
          final_cost -= (n_dir - 1) * unary

        if enable_norm_unary:
          min_value, _ = final_cost.min(2, keepdim=True)
          final_cost -= min_value

        # Naive combination of SGM
        for i in range(1, n_iter):
          if i % 10 == 0:
            print('image: {}, dir: {}, iter: {}'.format(img_name, n_dir, i))

          unary_new = final_cost.clone()
          final_cost, _ = mp_module(unary_new)

          if enable_minus_unary:
            final_cost -= (n_dir - 1) * unary_new

          # Avoid overflow
          if enable_norm_unary:
            min_value, _ = final_cost.min(2, keepdim=True)
            final_cost -= min_value

        final_seg = final_cost.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        final_seg = np.expand_dims(np.argmin(final_seg, 2).astype(np.int8), 2)
        print('image: {}, dir: {}, time: {:.4f}' \
              .format(img_name, n_dir, time.time() - time_start))

        scio.savemat(file_path,
                     {'n_iter': n_iter,
                      'n_dir': n_dir,
                      'rho': rho,
                      'p_func': context,
                      'n_disp': n_disp,
                      'p_weight': p_weight,
                      'trunct': truncated,
                      'min_a_dir': False,
                      'unary': unary.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                      'label_context': smoothness_context.permute(1, 2, 0).detach().cpu().numpy(),
                      'seg_all': final_seg})