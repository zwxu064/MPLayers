import torch
import os
import time
import copy
from message_passing import MessagePassingModule
from MP_module import MPModule
from MP_module_manual import test_mp_module_manual

ENABLE_DEBUG = True

def test_mp_module(mode, n_iter, unary, n_dir, label_context=None,
                   edge_weight=None, enable_message_vector=False,
                   enable_soft_weight=False, enable_debug=False):
  assert mode in ['ISGMR', 'TRWP']
  forward_time, backward_time = 0, 0
  enable_parallel = True if mode == 'ISGMR' else False
  batch, n_labels, h, w = unary.size()
  mp_module = MessagePassingModule(max_iter=n_iter, n_labels=n_labels,
                                   n_dirs=n_dir, mode=mode,
                                   ISGMR_parallel=enable_parallel,
                                   enable_debug=False,
                                   target='Stereo', graph_model='min_sum',
                                   label_context=label_context,
                                   n_edge_feats=1, llambda=1.,
                                   enable_Gaussian=False,
                                   enable_message_vector=enable_message_vector,
                                   enable_soft_weight=enable_soft_weight,
                                   enable_loopy=enable_debug)
  mp_module = mp_module.to(unary.device)

  unary_score = -unary
  img = torch.randint(0, 1, (batch, 3, h, w), dtype=torch.float32)
  img = img.to(unary.device)

  if enable_backward:
    mp_module.eval()
  else:
    mp_module.train()

  torch.cuda.synchronize() if enable_cuda else None
  time_start = time.time()
  unary_prob, pairwise_prob, message, message_init, label_context, \
    message_vector, message_index = mp_module.forward(unary_score, img,
                                                      edge_weight)
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

    return unary_prob, message.permute(1, 0, 2, 3, 4), \
           unary.grad, message_init.grad.permute(1, 0, 2, 3, 4), \
           label_context.grad, edge_weight.grad, forward_time, backward_time, \
           message_vector, message_index
  else:
    return unary_prob, message.permute(1, 0, 2, 3, 4), None, None, None, None, \
           forward_time, backward_time, message_vector, message_index


if __name__ == '__main__':
  # Note CPU has no problem, CUDA auto grad is probably wrong,
  # screenshot CPU results and compare with CUDA, will find the problem;
  # manual grad has no problem but need to use min-norm, otherwise overflow
  os.environ["CUDA_VISIBLE_DEVICES"] = "3"
  enable_cuda = torch.cuda.is_available()
  device = torch.device('cuda' if enable_cuda else 'cpu')
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  enable_backward = True

  mode, enable_soft = 'TRWP', True
  repeats_cuda, n_iter, n_dir, n_disp, h, w, n_cv = 1, 5, 4, 21, 128, 128, 1
  manual_thre = 1
  rho = 0.5 if (mode == 'TRWP') else 1
  rho = 1 if ENABLE_DEBUG else rho
  enable_my_cuda = True

  # Note: TODO (done and mark here)
  # Issue 1: do not switch on this, because cuda return argmin index unstably for same values
  # This is important for unittest but not affect the training.
  # CPU is always return high index for 1.1.0, and low index for 0.4.1
  # but GPU is unstable, try
  # => xx = torch.tensor([1, 2, 2, 0, 0,0,  2]).cuda()
  # => xx1 = torch.tensor([1, 2, 2, 0, 0, 0, 2]).cpu()
  # => print(xx.argmin(), xx1.argmin())
  # Issue 2: exp() will have precision error, making index unnitest failed.
  # Hence,
  # switch on ENABLE_DEBUG to set an external msg_index from Auto-grad version for debugging
  enable_auto_cuda = False

  assert n_disp <= 64
  repeats_auto = 0
  enable_message_vector = True
  seed = 2019

  # ==== random data =================
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

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
      dlabel_context_ref, dedge_weight_ref, forward_time, backward_time, \
      message_vector_ref, message_index_ref = \
      test_mp_module(mode, n_iter, unary, n_dir,
                     label_context=label_context,
                     edge_weight=edge_weight,
                     enable_message_vector=enable_message_vector,
                     enable_soft_weight=enable_soft,
                     enable_debug=ENABLE_DEBUG)

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
      test_mp_module_manual(n_dir, n_iter, unary.squeeze(0), label_context,
                            rho, enable_backward)

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
                         enable_soft=enable_soft)

    torch.cuda.synchronize() if enable_cuda else None
    create_time = time.time() - time_start

    if enable_backward:
      mp_module.train()
    else:
      mp_module.eval()

    # For debugging
    msg_norm_indx_in = message_index_ref if (ENABLE_DEBUG and repeats_auto > 0) else None
    msg_norm_indx_in = msg_norm_indx_in.permute(0, 1, 3, 2, 4, 5) if (n_cv > 0 and msg_norm_indx_in is not None) else None
    msg_norm_indx_in = msg_norm_indx_in.cuda().contiguous() if msg_norm_indx_in is not None else None

    torch.cuda.synchronize() if enable_cuda else None
    time_start = time.time()

    cost_final6, label_context6, _, message_vector_cuda, message_index_cuda = \
      mp_module(unary, edge_weight, msg_norm_index=msg_norm_indx_in)

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
          .format(create_time, forward_time, backward_time,
                  create_time + forward_time + backward_time))
    print('final cost sum, cuda: {}' \
          .format(cost_final6.abs().cpu().sum())) if repeats_cuda != 1 else None
    time_all += create_time + forward_time + backward_time
    torch.cuda.empty_cache()

  if repeats_cuda > 0:
    print('cuda average time {:.4f}s, forward {:.4f}s, backward {:.4f}s' \
          .format(time_all / repeats_cuda,
                  time_all_forward / repeats_cuda,
                  time_all_backward / repeats_cuda))

  if (repeats_cuda == 1) and (repeats_auto > 0):
    # CPU and GPU precisions are different, so use CPU or GPU for both
    print('final cost check, ref: {}, cuda: {}, diff: {}'
          .format(unary_final_ref.cpu().sum(), cost_final6.cpu().sum(),
                  (unary_final_ref.cpu() - cost_final6.cpu()).abs().max()))

    if enable_soft:
      print('message vector, ref: {}, cuda: {}, diff: {}' \
            .format(message_vector_ref.double().sum(),
                    message_vector_cuda.cpu().double().sum(),
                    (message_vector_ref.cpu() - message_vector_cuda.transpose(2, 3).cpu()).abs().max()))  # TODO

    print('message index, ref: {}, cuda: {}, diff: {}'. \
          format(message_index_ref.double().sum(),
                 message_index_cuda.cpu().double().sum(),
                 (message_index_ref.cpu().int() - message_index_cuda.cpu().int()).abs().max()))
    # print('non zero list: {}'.format((message_index_ref.int() - message_index_cuda.cpu().int()).nonzero()))
    # message index:(n_iter,n_dir,batch,cv,h,w)
    # print(message_index_ref[:,0,0,0,19,47], message_index_cuda[:,0,0,0,19,47])

    for tree in range(h):
      for loc in range(w):
        value_ref = unary_final_ref[:,:,tree,loc].abs().cpu()
        value6 = cost_final6[:,:,:,tree,loc].abs().cpu()
        diff = value_ref.squeeze() - value6.squeeze()

    if enable_backward:
      print('dunary, ref: {}, cuda: {}, diff: {}' \
            .format(dunary_final_ref.double().abs().sum().cpu(),
                    unary.grad.double().abs().sum().cpu(),
                    (dunary_final_ref.cpu() - unary.grad.cpu()).abs().max()))
      print('dcontext, ref: {}, cuda: {}, diff: {}' \
            .format(dlabel_context_ref.double().abs().sum().cpu(),
                    mp_module.label_context.grad.double().abs().sum().cpu(),
                    (dlabel_context_ref.squeeze().cpu() - mp_module.label_context.grad.cpu()).abs().max()))
      print('dedge_weights, ref: {}, cuda: {}, diff: {}' \
            .format(dedge_weight_ref.double().abs().sum().cpu(),
                    edge_weight.grad.double().abs().sum().cpu(),
                    (dedge_weight_ref.cpu() - edge_weight.grad.cpu()).abs().max()))

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
