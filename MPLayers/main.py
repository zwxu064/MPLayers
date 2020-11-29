import torch
import os
import time
import sys
import copy
import scipy.io as scio
import numpy as np
import argparse
sys.path.append('../pytorch')
from message_passing import MessagePassingModule
from MP_module import MPModule
from MP_module_manual import test_mp_module_manual
from test_compute_terms import MRFParams, compute_terms_py


os.environ["CUDA_VISIBLE_DEVICES"]="3"
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
enable_cuda = torch.cuda.is_available()
device = 'cpu'  # torch.device('cuda' if enable_cuda else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
enable_backward = False


def get_seg_all_iter(cost_all):
  seg_all = []

  for idx in range(cost_all.size(0)):
    cost = cost_all[idx].squeeze().detach().cpu().numpy()
    seg_all.append(np.argmin(cost, 2).astype(np.int8))  # numpy arg returns min indices

  return np.stack(seg_all, 2)


def test_mp_module(mode, n_iter, unary, n_dir, label_context):
  assert mode in ['ISGMR', 'TRWP']
  enable_parallel = True if mode == 'ISGMR' else False
  batch, n_disp, h, w = 1, unary.size(0), unary.size(1), unary.size(2)
  mp_module = MessagePassingModule(max_iter=n_iter, n_labels=n_disp, n_dirs=n_dir, mode=mode,
                                   ISGMR_parallel=enable_parallel, enable_debug=False,
                                   target='Stereo', graph_model='min_sum',
                                   label_context=label_context, n_edge_feats=1, llambda=1.,
                                   enable_Gaussian=False)
  mp_module = mp_module.to(device)

  unary_score = -unary.unsqueeze(0)
  img = torch.randint(0, 1, (batch, 3, h, w), dtype=torch.float32, device=device)
  unary_prob, pairwise_prob, message, message_init, label_context = mp_module.forward(unary_score, img)

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
  parser = argparse.ArgumentParser(description='Funny Witch')
  parser.add_argument('--server', type=str, default='data61', choices={'data61', '039614'})
  parser.add_argument('--img_name', type=str, default='tsukuba')
  parser.add_argument('--mode', type=str, default='TRWP')
  parser.add_argument('--n_dir', type=int, default=4)
  parser.add_argument('--n_iter', type=int, default=50)
  parser.add_argument('--context', type=str, default='TL')
  parser.add_argument('--rho', type=float, default=None)
  parser.add_argument('--enable_min_a_dir', action='store_true', default=False)
  parser.add_argument('--enable_saving_label', action='store_true', default=False)
  parser.add_argument('--enable_cuda_sgm', action='store_true', default=False)
  parser.add_argument('--truncated', type=int, default=1)
  parser.add_argument('--n_disp', type=int, default=10)
  parser.add_argument('--p_weight', type=int, default=10)
  args = parser.parse_args()
  img_name = args.img_name
  n_dir = args.n_dir
  n_iter = args.n_iter
  enable_min_a_dir = args.enable_min_a_dir != 0
  enable_saving_label = args.enable_saving_label != 0
  rho = args.rho
  mode = args.mode
  context = args.context
  truncated = args.truncated
  n_disp = args.n_disp
  p_weight = args.p_weight
  grad_thresh, grad_penalty = 0, 0

  # Note: one iter CUDA SGM and CPU SGM are the same, but iterative versions are different
  # iterative CUDA SGM is better cuz it does not aggregate after every iteration but after all iterations
  # while iterative CPU SGM aggregates after every iteration
  if args.enable_cuda_sgm:
    assert n_iter == 1

  if rho is None:
    rho = 0.5 if (mode == 'TRWP') else 1

  assert n_disp <= 192

  # Compute terms
  if img_name[0:3] == '000':
    left_img_path = '../data/{}/KITTI2015/image_2/{}.png'.format(args.server, img_name)
    right_img_path = '../data/{}/KITTI2015/image_3/{}.png'.format(args.server, img_name)
  elif img_name.split('_')[-1][-2:] in ['1l', '2l', '3l', '1s', '2s', '3s']:
    left_img_path = '../data/{}/ETH3D/training/{}/im0.png'.format(args.server, img_name)
    right_img_path = '../data/{}/ETH3D/training/{}/im1.png'.format(args.server, img_name)
  else:
    postfix = 'pgm' if img_name == 'map' else 'ppm'
    left_img_path = '../data/{}/Middlebury/middlebury/{}/imL.{}'.format(args.server, img_name, postfix)
    right_img_path = '../data/{}/Middlebury/middlebury/{}/imR.{}'.format(args.server, img_name, postfix)

  # ==== Save file path
  if enable_saving_label:
    save_dir = '../exp/{}'.format(img_name)
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)

    file_path = os.path.join(save_dir, '{}_{}_iter_{}_{}_trunc_{}_dir_{}_rho_{}' \
                             .format(img_name, mode, n_iter, context, truncated,
                                     n_dir, rho))
    file_path = file_path + '_minAdir' if enable_min_a_dir else file_path
    file_path = file_path + '.mat'

    if os.path.exists(file_path):
      print('{} exist, exit.'.format(file_path))
      exit()
  else:
    file_path = None

  # ==== Get terms
  param = MRFParams(left_img_path, right_img_path, context, n_disp, grad_thresh,
                    grad_penalty, truncated)
  data_cost, RGB, smoothness_context, param = compute_terms_py(param)
  smoothness_context *= p_weight

  # ==== Inference
  h, w, = param.height, param.width
  repeats, n_cv, manual_thre = 1, 1, 1

  # ==== Auto
  smoothness_context = smoothness_context.view(1,n_disp,n_disp).repeat(n_dir, 1, 1)
  label_context = smoothness_context.contiguous()
  unary = data_cost.permute(2, 0, 1).contiguous()

  unary_auto = copy.deepcopy(unary)
  unary_auto.requires_grad = True if enable_backward else False
  unary_auto = unary_auto.to(device)
  label_context_auto = copy.deepcopy(label_context).to(device)

  torch.cuda.synchronize()
  time_start = time.time()
  if (repeats == 1) and (not enable_saving_label):
    assert n_cv == 1
    unary_final_ref, message_final_ref, dunary_final_ref, dmessage_final_ref, \
      dlabel_context_ref = test_mp_module(mode, n_iter, unary_auto, n_dir,
                                          label_context_auto)
    unary_final4 = unary_final_ref
    message_final4 = message_final_ref
    dunary_final4 = dunary_final_ref
    dmessage_final4 = dmessage_final_ref
    dlabel_context4 = dlabel_context_ref
  torch.cuda.synchronize()
  print('pytorch auto time: {:.4f} s'.format(time.time() - time_start))

  # ==== Manual
  unary_manual = copy.deepcopy(unary)
  unary_manual.requires_grad = True if enable_backward else False
  unary_manual = unary_manual.view(1,1,n_disp,h,w).to(device)
  label_context_manual = copy.deepcopy(label_context).to(device)

  torch.cuda.synchronize()
  time_start = time.time()
  if (repeats == 1) and (h * w <= manual_thre) and (mode == 'ISGMR'):
    assert n_cv == 1
    unary_final_ref, message_final_ref, msg_indx_ref, msg_norm_indx_ref, \
      dunary_final_ref, dmessage_final_ref, dlabel_context_ref = \
      test_mp_module_manual(n_dir, n_iter, unary_manual, label_context_manual,
                            rho, enable_backward)
    unary_final5 = unary_final_ref
    message_final5 = message_final_ref
    dunary_final5 = dunary_final_ref
    dmessage_final5 = dmessage_final_ref
    dlabel_context5 = dlabel_context_ref
  torch.cuda.synchronize()
  print('pytorch manual time: {:.4f} s'.format(time.time() - time_start))

  # ==== CUDA
  time_all, time_all_forward, time_all_backward = 0, 0, 0
  for idx in range(repeats):
    torch.cuda.empty_cache()
    unary_cuda = copy.deepcopy(unary)
    unary_cuda.requires_grad = True if enable_backward else False
    unary_cuda = unary_cuda.view(1, 1, n_disp,h,w).to('cuda')
    label_context_cuda = copy.deepcopy(label_context).to('cuda')

    if mode == 'SGM':  # TODO
      unary_cuda = unary_cuda.cpu()
      label_context_cuda = label_context_cuda.cpu()

    torch.cuda.synchronize()
    time_start = time.time()

    if False:
      mp_module = MPModule(n_dir=n_dir,
                           n_iter=n_iter,
                           n_disp=n_disp,
                           mode=mode, rho=rho,
                           label_context=label_context_cuda,
                           enable_saving_label=enable_saving_label,
                           enable_min_a_dir=enable_min_a_dir,
                           term_weight=5)
    else:
      args.mpnet_mrf_mode = mode
      args.n_classes = n_disp
      args.mpnet_smoothness_train = ''
      args.mpnet_term_weight = 5
      args.mpnet_n_dirs = n_dir
      args.mpnet_max_iter = n_iter
      args.rho = rho
      mp_module = MPModule(args,
                           enable_create_label_context=False,
                           enable_saving_label=enable_saving_label,
                           enable_min_a_dir=enable_min_a_dir)
      mp_module.set_label_context(label_context_cuda, None, None)

    torch.cuda.synchronize()
    create_time = time.time() - time_start

    if enable_backward:
      mp_module.train()
    else:
      mp_module.eval()

    torch.cuda.synchronize()
    time_start = time.time()
    results = mp_module(unary_cuda)
    cost_final6, cost_all = results[0], results[2]
    cost_final6 = -cost_final6  # convert to fake prob
    torch.cuda.synchronize()
    forward_time = time.time() - time_start
    time_all_forward += forward_time

    torch.cuda.synchronize()
    time_start = time.time()

    if enable_backward:
      loss = cost_final6.sum()
      mp_module.label_context.retain_grad()
      unary_cuda.retain_grad()
      loss.backward()

    torch.cuda.synchronize()
    backward_time = time.time() - time_start
    time_all_backward += backward_time
    print(idx, 'cuda time, create: {:.4f} s, forward: {:.4f} s, backward: {:.4f} s, all: {:.4f} s' \
          .format(create_time, forward_time, backward_time, create_time + forward_time + backward_time))
    print('final cost sum, cuda: {}'.format(cost_final6.abs().cpu().sum())) if repeats != 1 else None
    time_all += create_time + forward_time + backward_time
    torch.cuda.empty_cache()

    # ==== Save for energy
    if enable_saving_label:
      if len(cost_all.size()) == 6:
        seg_all = get_seg_all_iter(cost_all)
      else:
        seg_all = cost_all

      scio.savemat(file_path, {'n_iter': n_iter,
                               'n_dir': n_dir,
                               'rho': rho,
                               'p_func': context,
                               'n_disp': n_disp,
                               'p_weight': p_weight,
                               'trunct': truncated,
                               'min_a_dir': int(enable_min_a_dir),
                               'unary': unary_cuda.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                               'label_context': smoothness_context.permute(1, 2, 0).detach().cpu().numpy(),
                               'seg_all': seg_all.squeeze(1).squeeze(1).permute(1, 2, 0).detach().cpu().numpy()})

    if (repeats > 1) and enable_backward:
      print(unary_cuda.grad.abs().mean().cpu().numpy(),
            mp_module.label_context.grad.abs().mean().cpu().numpy())
      print(unary_cuda.grad.abs().long().sum().cpu().numpy(),
            mp_module.label_context.grad.long().abs().sum().cpu().numpy())

    if idx != repeats - 1:
      del unary_cuda, label_context_cuda

  print('cuda average time {:.4f}s, forward {:.4f}s, backward {:.4f}s' \
        .format(time_all / repeats, time_all_forward / repeats, time_all_backward / repeats))

  if (repeats == 1) and (not enable_saving_label):
    # CPU and GPU precisions are different, so use CPU or GPU for both
    print('final cost check, ref: {}, cuda: {}'
          .format(unary_final_ref.cpu().sum(), cost_final6.cpu().sum()))

    # if (h * w <= manual_thre):
    #   print('msg_indx check, ref: {}, cuda: {}' \
    #         .format(msg_indx_ref.abs().cpu().sum(), msg_indx6.abs().cpu().sum()))
    #   print('msg_norm_indx check, ref: {}, cuda: {}' \
    #         .format(msg_norm_indx_ref.abs().cpu().sum(), msg_norm_indx6.abs().cpu().sum()))

    if enable_backward:
      print('ref, dunary:', dunary_final_ref.long().sum().cpu().numpy(),
            ', dcontext:', dlabel_context_ref.long().sum().cpu().numpy())
      print('cuda, dunary:', unary_cuda.grad.long().sum().cpu().numpy(),
            ', dcontext:', mp_module.label_context.grad.long().sum().cpu().numpy())

  # ==============
  if (repeats == 1) and (h * w <= manual_thre) and enable_backward:
    unary_abs_diff = (unary_final4.squeeze()-unary_final5.squeeze()).abs()
    message_abs_diff = (message_final4.squeeze()-message_final5.squeeze()).abs()
    dunary_abs_diff = (dunary_final4.squeeze()-dunary_final5.squeeze()).abs()
    dmessage_abs_diff = (dmessage_final4.squeeze()-dmessage_final5.squeeze()).abs()
    dlabel_context_abs_diff = (dlabel_context4.squeeze() - dlabel_context5.squeeze()).abs()
    print('===> pytorch auto and manual check, unary: {}, message: {}, dunary: {},'
          ' dmessage: {}, dlabel_context: {}' \
          .format(unary_abs_diff.sum(), message_abs_diff.sum(), dunary_abs_diff.sum(),
                  dmessage_abs_diff.sum(), dlabel_context_abs_diff.sum()))
