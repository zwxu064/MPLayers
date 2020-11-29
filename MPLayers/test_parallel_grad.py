import torch
import os
import time
import sys
import copy
sys.path.append('../pytorch')
from message_passing import MessagePassingModule
from MP_module import MPModule
from MP_module_manual import test_mp_module_manual
# from lib_stereo_slim import ISGMR

os.environ["CUDA_VISIBLE_DEVICES"]="3"
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
enable_cuda = torch.cuda.is_available()
device = torch.device('cuda' if enable_cuda else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
enable_backward = True


def isgmr_auto_grad(n_iter, rho, unary, message):
  message_update = []
  message_update.append(message)
  for iter in range(n_iter):
    message_one = message_update[-1]
    message_lr = rho * (unary + message_one.sum(0)) - message_one[1,:,:,:]
    message_rl = rho * (unary + message_one.sum(0)) - message_one[0,:,:,:]
    message_update.append(torch.stack((message_lr, message_rl), dim=0))

  loss = message_update[-1].sum()
  loss.backward()
  print('isgmr grad: {}\n{}'.format(unary.grad, message.grad))


def isgmr_auto_grad_expand(n_iter, rho, unary, message):
  message_lr = rho * (unary + message.sum(0)) - message[1,:,:,:]
  message_rl = rho * (unary + message.sum(0)) - message[0,:,:,:]
  message_update = torch.stack((message_lr, message_rl), dim=0)

  message_lr = rho * (unary + message_update.sum(0)) - message_update[1,:,:,:]
  message_rl = rho * (unary + message_update.sum(0)) - message_update[0,:,:,:]
  message_final = torch.stack((message_lr, message_rl), dim=0)

  message_update.retain_grad()
  message_final.retain_grad()
  loss = message_final.sum()
  loss.backward()
  print('isgmr grad expand: {}\n{}\n{}\n{}\n{}'.format(message_final, unary.grad,
                                                       message.grad, message_update.grad, message_final.grad))


def trwp_auto_grad(n_iter, rho, unary, message):
  message_update = message
  for iter in range(n_iter):
    message_update = message_update.clone()
    message_update[0,:,:,:] = rho * (unary + message_update.sum(0)) - message_update[1,:,:,:]
    message_update[1,:,:,:] = rho * (unary + message_update.sum(0)) - message_update[0,:,:,:]

  loss = message_update.sum()
  loss.backward()
  print('trwp grad: {}\n{}'.format(unary.grad, message.grad))


def test_mp_module(mode, n_iter, unary, n_dir, label_context=None, edge_weight=None):
  assert mode in ['ISGMR', 'TRWP']
  forward_time, backward_time = 0, 0
  enable_parallel = True if mode == 'ISGMR' else False
  batch, n_labels, h, w = unary.size()
  mp_module = MessagePassingModule(max_iter=n_iter, n_labels=n_labels, n_dirs=n_dir, mode=mode,
                                   ISGMR_parallel=enable_parallel, enable_debug=False,
                                   target='Stereo', graph_model='min_sum',
                                   label_context=label_context, n_edge_feats=1, llambda=1.,
                                   enable_Gaussian=False)
  mp_module = mp_module.to(unary.device)

  unary_score = -unary
  img = torch.randint(0, 1, (batch, 3, h, w), dtype=torch.float32).to(unary.device)

  if enable_backward:
    mp_module.eval()
  else:
    mp_module.train()

  torch.cuda.synchronize() if enable_cuda else None
  time_start = time.time()
  unary_prob, pairwise_prob, message, message_init, label_context = \
    mp_module.forward(unary_score, img, edge_weight)
  torch.cuda.synchronize() if enable_cuda else None
  forward_time = time.time() - time_start

  if enable_backward:
    loss = unary_prob.sum()
    unary.retain_grad()  # for CUDA
    label_context.retain_grad()
    edge_weight.retain_grad()

    torch.cuda.synchronize() if enable_cuda else None
    time_start = time.time()
    loss.backward()
    backward_time = time.time() - time_start

    return unary_prob, message.permute(1,0,2,3,4), \
           unary.grad, message_init.grad.permute(1,0,2,3,4), label_context.grad, \
           edge_weight.grad, forward_time, backward_time
  else:
    return unary_prob, message.permute(1,0,2,3,4), None, None, None, \
           forward_time, backward_time


if __name__ == '__main__':
  # Note CPU has no problem, CUDA auto grad is probably wrong,
  # screenshot CPU results and compare with CUDA, will find the problem;
  # manual grad has no problem but need to use min-norm, otherwise overflow
  mode = 'TRWP'
  repeats_cuda, n_iter, n_dir, n_disp, h, w, n_cv = 1, 5, 8, 48, 25, 36, 1
  manual_thre = 1
  rho = 0.5 if (mode == 'TRWP') else 1
  enable_my_cuda = True
  enable_auto_cuda = False
  assert n_disp <= 192
  repeats_auto = 0

  # message(dir,disp,h,w), unary(disp,h,w)
  # torch.manual_seed(2019)
  # torch.cuda.manual_seed_all(2019)
  # unary = torch.randint(0, 5, (n_disp, h, w), dtype=torch.float32, device=device, requires_grad=True)
  # message = torch.zeros(n_dir, n_disp, h, w, dtype=torch.float32, device=device, requires_grad=True)
  #
  # torch.manual_seed(2019)
  # torch.cuda.manual_seed_all(2019)
  # unary2 = torch.randint(0, 5, (n_disp, h, w), dtype=torch.float32, device=device, requires_grad=True)
  # message2 = torch.zeros(n_dir, n_disp, h, w, dtype=torch.float32, device=device, requires_grad=True)
  #
  # torch.manual_seed(2019)
  # torch.cuda.manual_seed_all(2019)
  # unary3 = torch.randint(0, 5, (n_disp, h, w), dtype=torch.float32, device=device, requires_grad=True)
  # message3 = torch.zeros(n_dir, n_disp, h, w, dtype=torch.float32, device=device, requires_grad=True)
  # print('input', unary3, '\n', message3)

  # isgmr_auto_grad(2, rho, unary, message)
  # trwp_auto_grad(2, rho, unary2, message2)
  # isgmr_auto_grad_expand(2, rho, unary3, message3)

  # ==== random data =================
  torch.manual_seed(2019)
  torch.cuda.manual_seed_all(2019)
  unary_org = torch.randint(0, 255, (n_cv, n_disp, h, w), dtype=torch.float32, device=device, requires_grad=True)
  label_context_org = torch.randint(0, n_disp, (n_dir, n_disp, n_disp), dtype=torch.float32, device=device, requires_grad=True)
  edge_weight_org = torch.randint(1, 5, (n_cv, n_dir, h, w), dtype=torch.float32, device=device, requires_grad=True)

  # ==== auto =================
  unary = copy.deepcopy(unary_org).contiguous()
  label_context = copy.deepcopy(label_context_org).contiguous()
  edge_weight = copy.deepcopy(edge_weight_org).contiguous()

  if enable_auto_cuda:
    unary = unary.cuda()
    label_context = label_context.cuda()
    edge_weight = edge_weight.cuda()
  else:
    unary = unary.cpu()
    label_context = label_context.cpu()
    edge_weight = edge_weight.cpu()

  print('PyTorch auto mode: {}, repeats: {}, dir: {}, size: {}*{}*{}' \
        .format(mode, repeats_auto, n_dir, h, w, n_disp))
  time_all, forward_time_all, backward_time_all = 0, 0, 0
  for idx in range(repeats_auto):
    unary_final_ref, message_final_ref, dunary_final_ref, dmessage_final_ref, \
      dlabel_context_ref, dedge_weight_ref, forward_time, backward_time = \
      test_mp_module(mode, n_iter, unary, n_dir, label_context=label_context,
                     edge_weight=edge_weight)
    unary_final4 = unary_final_ref
    message_final4 = message_final_ref
    dunary_final4 = dunary_final_ref
    dmessage_final4 = dmessage_final_ref
    dlabel_context4 = dlabel_context_ref
    forward_time_all += forward_time
    backward_time_all += backward_time
    time_all += forward_time + backward_time
    print('PyTorch index: {}, {:.4f}s, forward: {:.4f}s, backward: {:.4f}s' \
          .format(idx, forward_time + backward_time, forward_time, backward_time))

  if repeats_auto > 0:
    print('pytorch auto average time: {:.4f}s, forward: {:.4f}s, backward: {:.4f}s' \
          .format(time_all / repeats_auto,
                  forward_time_all / repeats_auto,
                  backward_time_all / repeats_auto))

  # ==== manual =================
  unary = copy.deepcopy(unary_org).contiguous().cpu()
  label_context = copy.deepcopy(label_context_org).contiguous().cpu()
  edge_weight = copy.deepcopy(edge_weight_org).contiguous().cpu()

  torch.cuda.synchronize() if enable_cuda else None
  time_start = time.time()
  if (repeats_cuda == 1) and (h * w <= manual_thre) and (mode == 'ISGMR'):
    assert n_cv == 1
    unary_final_ref, message_final_ref, msg_indx_ref, msg_norm_indx_ref, \
      dunary_final_ref, dmessage_final_ref, dlabel_context_ref = \
      test_mp_module_manual(n_dir, n_iter, unary.squeeze(0), label_context, rho, enable_backward)
    unary_final_ref = unary_final_ref.unsqueeze(0)
    message_final_ref = message_final_ref.unsqueeze(0)
    dunary_final_ref = dunary_final_ref.unsqueeze(0)
    dmessage_final_ref = dmessage_final_ref.unsqueeze(0)
    dlabel_context_ref = dlabel_context_ref.unsqueeze(0)

    unary_final5 = unary_final_ref
    message_final5 = message_final_ref
    dunary_final5 = dunary_final_ref
    dmessage_final5 = dmessage_final_ref
    dlabel_context5 = dlabel_context_ref
  torch.cuda.synchronize() if enable_cuda else None
  print('pytorch manual time: {:.4f} s'.format(time.time() - time_start))

  # ==== cuda =================
  time_all, time_all_forward, time_all_backward = 0, 0, 0
  print('CUDA repeats: {}, dir: {}, size: {}*{}*{}' \
        .format(repeats_cuda, n_dir, h, w, n_disp))
  for idx in range(repeats_cuda):
    torch.cuda.empty_cache()
    unary = copy.deepcopy(unary_org).unsqueeze(0).contiguous()
    label_context = copy.deepcopy(label_context_org).contiguous()
    # edge_weight = copy.deepcopy(edge_weight_org).unsqueeze(0).contiguous()
    edge_weight = copy.deepcopy(edge_weight_org).contiguous()

    if enable_my_cuda:
      unary = unary.cuda()
      label_context = label_context.cuda()
      edge_weight = edge_weight.cuda()
    else:
      unary = unary.cpu()
      label_context = label_context.cpu()
      edge_weight = edge_weight.cpu()

    torch.cuda.synchronize() if enable_cuda else None
    time_start = time.time()
    mp_module = MPModule(n_dir=n_dir,
                         n_iter=n_iter,
                         n_disp=n_disp,
                         mode=mode,
                         rho=rho,
                         enable_cuda=enable_my_cuda,
                         label_context=None,
                         smoothness_train='ON',
                         smoothness_mode='TQ')
    torch.cuda.synchronize() if enable_cuda else None
    create_time = time.time() - time_start

    if enable_backward:
      mp_module.train()
    else:
      mp_module.eval()

    torch.cuda.synchronize() if enable_cuda else None
    time_start = time.time()
    cost_final6, _, _ = mp_module(unary, edge_weight)
    cost_final6 = -cost_final6  # convert to fake prob
    torch.cuda.synchronize() if enable_cuda else None
    forward_time = time.time() - time_start
    time_all_forward += forward_time

    torch.cuda.synchronize() if enable_cuda else None
    time_start = time.time()

    if enable_backward:
      loss = cost_final6.sum()
      mp_module.label_context.retain_grad()
      unary.retain_grad()
      edge_weight.retain_grad()
      loss.backward()

    torch.cuda.synchronize() if enable_cuda else None
    backward_time = time.time() - time_start
    time_all_backward += backward_time
    print(idx, 'cuda time, create: {:.4f} s, forward: {:.4f} s, backward: {:.4f} s, all: {:.4f} s' \
          .format(create_time, forward_time, backward_time, create_time + forward_time + backward_time))
    print('final cost sum, cuda: {}'.format(cost_final6.abs().cpu().sum())) if repeats_cuda != 1 else None
    time_all += create_time + forward_time + backward_time
    torch.cuda.empty_cache()

    # if (repeats_cuda > 1) and enable_backward:
    #   print(unary6.grad.abs().mean().cpu().numpy(),
    #         mp_module.label_context.grad.abs().mean().cpu().numpy())
    #   print(unary6.grad.abs().double().sum().cpu().numpy(),
    #         mp_module.label_context.grad.double().abs().sum().cpu().numpy())

  print(mp_module.label_context, '\n', mp_module.label_context.grad)

  if repeats_cuda > 0:
    print('cuda average time {:.4f}s, forward {:.4f}s, backward {:.4f}s' \
          .format(time_all / repeats_cuda,
                  time_all_forward / repeats_cuda,
                  time_all_backward / repeats_cuda))

  if (repeats_cuda == 1) and ((repeats_auto > 0) or ((h * w <= manual_thre) and (mode == 'ISGMR'))):
    # CPU and GPU precisions are different, so use CPU or GPU for both
    print('final cost check, ref: {}, cuda: {}'
          .format(unary_final_ref.cpu().sum(), cost_final6.cpu().sum()))

    # if (h * w <= manual_thre):
    #   print('msg_indx check, ref: {}, cuda: {}'.format(msg_indx_ref.abs().cpu().sum(), msg_indx6.abs().cpu().sum()))
    #   print('msg_norm_indx check, ref: {}, cuda: {}'.format(msg_norm_indx_ref.abs().cpu().sum(), msg_norm_indx6.abs().cpu().sum()))

    for tree in range(h):
      for loc in range(w):
        value_ref = unary_final_ref[:,:,tree,loc].abs().cpu()
        value6 = cost_final6[:,:,:,tree,loc].abs().cpu()
        diff = value_ref.squeeze() - value6.squeeze()
        if (rho != 1 and diff.max() > 1e-4) or (rho == 1 and diff.sum() != 0):
          print('wrong', tree, loc, value_ref.sum(), value6.sum())

    if enable_backward:
      print('ref, dunary:', dunary_final_ref.double().abs().sum().cpu().numpy(),
            ', dcontext:', dlabel_context_ref.double().abs().sum().cpu().numpy(),
            ', dedge_weights:', dedge_weight_ref.double().abs().sum().cpu().numpy())
      print('cuda, dunary:', unary.grad.double().abs().sum().cpu().numpy(),
            ', dcontext:', mp_module.label_context.grad.double().abs().sum().cpu().numpy(),
            ', dedge_weights:', edge_weight.grad.double().abs().sum().cpu().numpy())

  # ==============
  if (repeats_cuda == 1) and (h * w <= manual_thre) and enable_backward:
    unary_abs_diff = (unary_final4.squeeze()-unary_final5.squeeze())
    message_abs_diff = (message_final4.squeeze()-message_final5.squeeze())
    dunary_abs_diff = (dunary_final4.squeeze()-dunary_final5.squeeze())
    dmessage_abs_diff = (dmessage_final4.squeeze()-dmessage_final5.squeeze())
    dlabel_context_abs_diff = (dlabel_context4.squeeze() - dlabel_context5.squeeze())
    print('===> pytorch auto and manual check, unary: {}, message: {}, dunary: {},'
          ' dmessage: {}, dlabel_context: {}' \
          .format(unary_abs_diff.sum(), message_abs_diff.sum(), dunary_abs_diff.sum(),
                  dmessage_abs_diff.sum(), dlabel_context_abs_diff.sum()))
    # dunary_area = dunary_abs_diff != 0
    # dmessage_area = dmessage_abs_diff != 0
    # dlabel_context_area = dlabel_context_abs_diff != 0
    # if dunary_area.sum() > 0:
    #   print('dunary', dunary_final4[dunary_area], '\n', dunary_final5[dunary_area])
    # if dmessage_area.sum() > 0:
    #   print('dmessage', dmessage_final4[dmessage_area], '\n', dmessage_final5[dmessage_area])
    # if dlabel_context_area.sum() > 0:
    #   print('dlabel_context', dlabel_context4[dlabel_context_area], '\n', dlabel_context5[dlabel_context_area])


  # ==============
  # msg_norm = torch.arange(65).view(1, -1).float().to('cuda')
  # msg_norm[0, 0] = 99
  # msg_norm[0, 3] = 0
  # msg_norm[0, 4] = 0
  # msg_norm[0, 31] = -7
  # msg_norm[0, 63] = -8
  # ISGMR.test_msg(msg_norm)

  # ==============
  # if enable_cuda:
  #   stream_data = torch.zeros(32, 50000, dtype=torch.float32, device='cuda')
  #   repeats = 100
  #
  #   for enable_multiple in [0, 1]:
  #     time_all = 0
  #     for repeat in range(repeats):
  #       torch.cuda.synchronize() if enable_cuda else None
  #       time_start = time.time()
  #       ISGMR.test_stream(enable_multiple, stream_data)
  #       torch.cuda.synchronize() if enable_cuda else None
  #       duration = time.time() - time_start
  #       time_all += duration
  #
  #       if repeat == repeats - 1:
  #         if enable_multiple == 0:
  #           stream_data_1stream = stream_data
  #         elif enable_multiple == 1:
  #           stream_data_multistream = stream_data
  #
  #       # print('Multi-stream time: {:.4f} s, enable mutliple: {}, value: {:.4f}' \
  #       #       .format(duration, enable_multiple, stream_data.abs().mean()))
  #     print('====> enable_multi {}, average time: {:.4f} s'.format(enable_multiple, time_all / repeats))
  #   assert (stream_data_1stream - stream_data_multistream).sum() == 0
