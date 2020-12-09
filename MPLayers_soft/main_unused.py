import torch
import os
import time
import copy
import argparse
import matplotlib.pyplot as plt
from pytorch.message_passing import MessagePassingModule
from pytorch.test_compute_terms import MRFParams, compute_terms_py


def test_mp_module(mode, n_iter, unary, n_dir, label_context, enable_soft_weight):
  assert mode in ['ISGMR', 'TRWP']
  enable_parallel = True if mode == 'ISGMR' else False
  batch, n_disp, h, w = 1, unary.size(0), unary.size(1), unary.size(2)
  mp_module = MessagePassingModule(max_iter=n_iter, n_labels=n_disp, n_dirs=n_dir, mode=mode,
                                   ISGMR_parallel=enable_parallel, enable_debug=False,
                                   target='Stereo', graph_model='min_sum',
                                   label_context=label_context, n_edge_feats=1, llambda=1.,
                                   enable_Gaussian=False, enable_soft_weight=enable_soft_weight)
  mp_module = mp_module.to(device)

  unary_score = -unary.unsqueeze(0)
  img = torch.randint(0, 1, (batch, 3, h, w), dtype=torch.float32, device=device)
  unary_prob, pairwise_prob, message, message_init, label_context, _, _ = mp_module.forward(unary_score, img)

  if enable_backward:
    loss = unary_prob.sum()
    unary.retain_grad()  # for CUDA
    label_context.retain_grad()
    loss.backward()
    return unary_prob.squeeze(0), message.permute(1,0,2,3,4).squeeze(1), \
           unary.grad.squeeze(0), message_init.grad.permute(1,0,2,3,4).squeeze(1), label_context.grad.squeeze()
  else:
    return unary_prob.squeeze(0), message.permute(1,0,2,3,4).squeeze(1), None, None, None


if __name__ == '__main__':
  torch.manual_seed(2019)
  torch.cuda.manual_seed_all(2019)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  enable_backward = False

  parser = argparse.ArgumentParser(description='Funny Witch')
  parser.add_argument('--img_name', type=str, default='tsukuba')
  parser.add_argument('--mode', type=str, default='TRWP')
  parser.add_argument('--n_dir', type=int, default=4)
  parser.add_argument('--n_iter', type=int, default=50)
  parser.add_argument('--context', type=str, default='TL')
  parser.add_argument('--rho', type=float, default=1)
  parser.add_argument('--enable_saving_label', action='store_true', default=False)
  parser.add_argument('--truncated', type=int, default=1)
  parser.add_argument('--n_disp', type=int, default=16)
  parser.add_argument('--p_weight', type=int, default=10)
  parser.add_argument('--enable_cuda', action='store_true', default=False)
  parser.add_argument('--enable_display', action='store_true', default=False)
  parser.add_argument('--enable_soft_weight', action='store_true', default=False)
  parser.add_argument('--left_img_path', type=str, default='')
  parser.add_argument('--right_img_path', type=str, default='')
  parser.add_argument('--save_dir', type=str, default='.')
  args = parser.parse_args()
  grad_thresh, grad_penalty = 0, 0
  # data_dir = 'data'
  enable_cuda = torch.cuda.is_available() and args.enable_cuda
  device = torch.device('cuda' if enable_cuda else 'cpu')

  if args.rho is None:
    args.rho = 0.5 if (args.mode == 'TRWP') else 1

  assert args.n_disp <= 192

  # Compute terms
  if False:
    if args.img_name[0:3] == '000':
      left_img_path = os.path.join(data_dir, 'KITTI2015/image_2/{}.png'.format(args.img_name))
      right_img_path = os.path.join(data_dir, 'KITTI2015/image_3/{}.png'.format(args.img_name))
    elif args.img_name.split('_')[-1][-2:] in ['1l', '2l', '3l', '1s', '2s', '3s']:
      left_img_path = os.path.join(data_dir, 'ETH3D/training/{}/im0.png'.format(args.img_name))
      right_img_path = os.path.join(data_dir, 'ETH3D/training/{}/im1.png'.format(args.img_name))
    else:
      postfix = 'pgm' if args.img_name == 'map' else 'ppm'
      left_img_path = os.path.join(data_dir, 'Middlebury/middlebury/{}/imL.{}'.format(args.img_name, postfix))
      right_img_path = os.path.join(data_dir, 'Middlebury/middlebury/{}/imR.{}'.format(args.img_name, postfix))
  else:
    left_img_path = args.left_img_path
    right_img_path = args.right_img_path
    assert os.path.exists(left_img_path), 'Left image {} not exist'.format(left_img_path)
    assert os.path.exists(right_img_path), 'Right image {} not exist'.format(right_img_path)

  # ==== Save file path
  if args.enable_saving_label:
    if False:
      save_dir = 'exp/{}'.format(args.img_name)
    else:
      save_dir = '{}/{}'.format(args.save_dir, args.img_name)

    if not os.path.exists(save_dir):
      os.mkdir(save_dir)

    file_path = os.path.join(save_dir, '{}_{}_iter_{}_{}_trunc_{}_dir_{}_rho_{}' \
                             .format(args.img_name, args.mode, args.n_iter, args.context,
                                     args.truncated, args.n_dir, args.rho))
    file_path_full = file_path + '.mat'

    print(file_path_full)
    if os.path.exists(file_path_full):
      print('{} exist, exit.'.format(file_path_full))
      exit()
  else:
    file_path = None

  # ==== Get terms
  param = MRFParams(left_img_path, right_img_path, args.context, args.n_disp,
                    grad_thresh, grad_penalty, args.truncated)
  data_cost, RGB, smoothness_context, param = compute_terms_py(param)
  smoothness_context *= args.p_weight

  # ==== Inference
  h, w, = param.height, param.width
  repeats, n_cv, manual_thre = 1, 1, 1

  print(args)

  # ==== Auto
  smoothness_context = smoothness_context.view(1, args.n_disp, args.n_disp).repeat(args.n_dir, 1, 1)
  label_context = smoothness_context.contiguous()
  unary = data_cost.permute(2, 0, 1).contiguous()

  unary_auto = copy.deepcopy(unary)
  unary_auto.requires_grad = True if enable_backward else False
  unary_auto = unary_auto.to(device)
  label_context_auto = copy.deepcopy(label_context).to(device)

  torch.cuda.synchronize()
  time_start = time.time()

  if (repeats == 1) and (not args.enable_saving_label):
    assert n_cv == 1
    unary_final_ref, message_final_ref, dunary_final_ref, dmessage_final_ref, \
      dlabel_context_ref = test_mp_module(args.mode, args.n_iter, unary_auto, args.n_dir,
                                          label_context_auto, args.enable_soft_weight)
    unary_final4 = unary_final_ref
    message_final4 = message_final_ref
    dunary_final4 = dunary_final_ref
    dmessage_final4 = dmessage_final_ref
    dlabel_context4 = dlabel_context_ref
  else:
    unary_final_ref = None

  torch.cuda.synchronize()
  torch.cuda.empty_cache()
  print('pytorch auto time: {:.4f} s'.format(time.time() - time_start))

  if args.enable_display and (unary_final_ref is not None):
    seg = torch.argmax(unary_final_ref, dim=0)
    plt.figure()
    plt.imshow(seg.cpu().numpy())
    plt.show()
