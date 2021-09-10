import torch, sys
import torch.nn as nn
import numpy as np
sys.path.append('MPLayers')
from lib_stereo import TRWP as TRWP_stereo
from lib_stereo import ISGMR as ISGMR_stereo
from lib_seg import TRWP as TRWP_seg
from lib_seg import ISGMR as ISGMR_seg
sys.path.append('..')
from utils.label_context import create_label_context


# references:
# http://www.benjack.io/2017/06/12/python-cpp-tests.html
# https://pytorch.org/tutorials/advanced/cpp_extension.html

class TRWPFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, unary, label_context, edge_weights, args):
    # to, unary:(batch,cv,h,w,disp),message:(dir,batch,cv,h,w,disp),label_context:(disp,disp) for seg and (disp) for stereo
    # edge_weights:(dir,batch,cv,h,w)
    batch, cv, h, w, n_disp = unary.size()
    rho, n_iter, n_dir, is_training = args.rho, args.mpnet_max_iter, args.mpnet_n_dirs, args.training
    TRWP = TRWP_seg if (n_disp == 21) else TRWP_stereo

    message = unary.new_zeros(n_dir, batch, cv, h, w, n_disp)
    cost_final = unary.new_zeros(batch, cv, h, w, n_disp)
    unary_update = unary.new_zeros(batch, cv, h, w, n_disp)

    if is_training:
      message_index = unary.new_zeros(n_iter, n_dir, batch, cv, h, w, n_disp, dtype=torch.uint8)
      cost_index = unary.new_zeros(n_iter, n_dir, batch, cv, h, w, dtype=torch.uint8)
    else:
      message_index = torch.empty(0, dtype=torch.uint8)
      cost_index = torch.empty(0, dtype=torch.uint8)

    if args.enable_saving_label:
      label_all = unary.new_zeros(n_iter, batch, cv, h, w, dtype=torch.uint8)
    else:
      label_all = torch.empty(0, dtype=torch.uint8)

    if edge_weights is None:
      edge_weights = unary.new_ones(n_dir, batch, cv, h, w)
      enable_edge_weights = False
    else:
      enable_edge_weights = True

    TRWP.forward(rho, int(n_iter), int(args.enable_min_a_dir), unary, label_context,
                 edge_weights, message, cost_final, message_index, cost_index,
                 unary_update, label_all)
    ctx.intermediate_results = rho, message_index, cost_index, label_context, edge_weights, enable_edge_weights

    del message, message_index, cost_index, unary_update, label_context, edge_weights
    return cost_final, label_all

  @staticmethod
  def backward(ctx, dcost_final, dmessage_all):
    rho, message_index, cost_index, label_context, edge_weights, enable_edge_weights = ctx.intermediate_results
    del ctx.intermediate_results

    dcost_final = dcost_final.contiguous()
    n_iter, n_dir, batch, cv, h, w, n_disp = message_index.size()
    TRWP = TRWP_seg if (n_disp == 21) else TRWP_stereo

    dunary = dcost_final.new_zeros(batch, cv, h, w, n_disp)
    dmessage = dcost_final.new_zeros(n_dir, batch, cv, h, w, n_disp)
    dunary_update = dcost_final.new_zeros(batch, cv, h, w, n_disp)
    dedge_weights = dcost_final.new_zeros(n_dir, batch, cv, h, w)

    enable_seg = (n_disp == 21)
    if enable_seg:
      dlabel_context = dcost_final.new_zeros(n_disp, n_disp)
    else:
      dlabel_context = dcost_final.new_zeros(n_disp)

    TRWP.backward(rho, label_context, edge_weights, dcost_final, message_index,
                  cost_index, dunary, dlabel_context, dedge_weights, dmessage,
                  dunary_update)

    del message_index, cost_index, label_context, edge_weights, dcost_final, \
      dmessage, dunary_update

    if not enable_edge_weights:
      dedge_weights = None

    return dunary, dlabel_context, dedge_weights, None


class ISGMRFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, unary, label_context, edge_weights, args):
    # to, unary:(batch,cv,h,w,disp),message:(dir,batch,cv,h,w,disp),label_context:(disp,disp) for seg and (disp) for stereo
    batch, cv, h, w, n_disp = unary.size()
    rho, n_iter, n_dir, is_training = args.rho, args.mpnet_max_iter, args.mpnet_n_dirs, args.training
    ISGMR = ISGMR_seg if (n_disp == 21) else ISGMR_stereo
    n_dir_msg = n_dir

    if args.sgm_single_mode >= 0:
      n_iter, n_dir, n_dir_msg = 1, 1, 2  # n_dir_msg=2 for dir_inv in param in CUDA

    message = unary.new_zeros(n_dir_msg, batch, cv, h, w, n_disp)
    cost_final = unary.new_zeros(batch, cv, h, w, n_disp)
    unary_tmp = unary.new_zeros(batch, cv, h, w, n_disp)
    message_tmp = unary.new_zeros(n_dir_msg, batch, cv, h, w, n_disp)

    if is_training:
      message_index = unary.new_zeros(n_iter, n_dir_msg, batch, cv, h, w, n_disp, dtype=torch.uint8)
      cost_index = unary.new_zeros(n_iter, n_dir, batch, cv, h, w, dtype=torch.uint8)
    else:
      message_index = torch.empty(0, dtype=torch.uint8)
      cost_index = torch.empty(0, dtype=torch.uint8)

    if args.enable_saving_label:
      label_all = unary.new_zeros(n_iter, batch, cv, h, w, dtype=torch.uint8)
    else:
      label_all = torch.empty(0, dtype=torch.uint8)

    if edge_weights is None:
      edge_weights = unary.new_ones(n_dir, batch, cv, h, w)
      enable_edge_weights = False
    else:
      enable_edge_weights = True

    ISGMR.forward(int(args.enable_sgm), int(args.sgm_single_mode), rho, int(n_iter), int(args.enable_min_a_dir),
                  unary, label_context, edge_weights, message, cost_final, message_index,
                  cost_index, unary_tmp, message_tmp, label_all)
    ctx.intermediate_results = args.enable_sgm, args.sgm_single_mode, rho, message_index, cost_index, label_context, \
                               edge_weights, enable_edge_weights

    del message, message_index, cost_index, unary_tmp, message_tmp, label_context, edge_weights
    return cost_final, label_all

  @staticmethod
  def backward(ctx, dcost_final, dmessage_all):
    enable_sgm, sgm_single_mode, rho, message_index, cost_index, label_context, edge_weights, enable_edge_weights \
      = ctx.intermediate_results
    del ctx.intermediate_results

    dcost_final = dcost_final.contiguous()
    n_iter, n_dir, batch, cv, h, w, n_disp = message_index.size()
    ISGMR = ISGMR_seg if (n_disp == 21) else ISGMR_stereo
    n_dir_msg = n_dir

    if sgm_single_mode >= 0:
      n_iter, n_dir, n_dir_msg = 1, 1, 2  # n_dir_msg=2 for dir_inv in param in CUDA

    dunary = dcost_final.new_zeros(batch, cv, h, w, n_disp)
    dmessage = dcost_final.new_zeros(n_dir_msg, batch, cv, h, w, n_disp)
    dunary_tmp = dcost_final.new_zeros(batch, cv, h, w, n_disp)
    dmessage_tmp = dcost_final.new_zeros(n_dir_msg, batch, cv, h, w, n_disp)
    dedge_weights = dcost_final.new_zeros(n_dir, batch, cv, h, w)

    enable_seg = (n_disp == 21)
    if enable_seg:
      dlabel_context = dcost_final.new_zeros(n_disp, n_disp)
    else:
      dlabel_context = dcost_final.new_zeros(n_disp)

    ISGMR.backward(int(enable_sgm), int(sgm_single_mode), rho, label_context, edge_weights, dcost_final, message_index,
                   cost_index, dunary, dlabel_context, dedge_weights, dmessage,
                   dunary_tmp, dmessage_tmp)

    del dcost_final, message_index, cost_index, label_context, edge_weights, \
      dmessage, dunary_tmp, dmessage_tmp

    if not enable_edge_weights:
      dedge_weights = None

    return dunary, dlabel_context, dedge_weights, None


class MPModule(torch.nn.Module):
  # def __init__(self, n_iter=5, n_dir=16, n_disp=32, mode='ISGMR', rho=1, term_weight=1,
  #              label_context=None, enable_saving_label=False, enable_min_a_dir=False,
  #              enable_cuda=False, smoothness_mode='TL', smoothness_trunct_loc=None,
  #              smoothness_trunct_value=None, smoothness_train=None, enable_soft=False):
  def __init__(self, args, enable_create_label_context=False, enable_saving_label=False, enable_min_a_dir=False):
    super(MPModule, self).__init__()
    self.args = args
    self.args.rho = 0.5 if (args.mpnet_mrf_mode == 'TRWP') else 1
    self.args.enable_saving_label = enable_saving_label
    self.args.enable_min_a_dir = enable_min_a_dir
    self.args.enable_sgm = args.mpnet_mrf_mode == 'SGM'
    self.args.enable_seg = True if (args.n_classes == 21) else False

    # TODO this term weight can come from edge weights
    if enable_create_label_context:
      self.create_label_context()

  def get_label_context(self):
    return self.label_context, self.label_context_loc, self.label_context_diag_loc

  def set_label_context(self, label_context, label_context_loc, label_context_diag_loc):
    self.label_context = label_context
    self.label_context_loc = label_context_loc
    self.label_context_diag_loc = label_context_diag_loc

  def create_label_context(self):
    self.label_context, self.label_context_loc, self.label_context_diag_loc = \
      create_label_context(self.args, enable_seg=self.args.enable_seg,
                           enable_symmetric=self.args.enable_symmetric)

  def forward(self, unary, edge_weights=None):
    # unary:(batch,cv,n_disp,h,w); label_context:(n_disp,n_disp) for seg and (n_disp) for stereo
    # edge_weights:(batch,n_dir,h,w) unsqueeze(1) to be (batch,cv,n_dir,h,w)
    unary = unary.permute(0, 1, 3, 4, 2).contiguous()
    edge_weights = edge_weights.unsqueeze(1).permute(2, 0, 1, 3, 4).contiguous() \
      if (edge_weights is not None) else edge_weights

    # Note: one iter CUDA SGM and CPU SGM are the same, but iterative versions are different
    # iterative CUDA SGM is better cuz it does not aggregate after every iteration but after all iterations
    # while iterative CPU SGM aggregates after every iteration
    if unary.is_cuda and (self.args.mpnet_mrf_mode == 'SGM'):
      assert self.args.mpnet_max_iter == 1

    label_context = self.label_context * 1
    if self.args.mpnet_smoothness_train == 'sigmoid':
      label_context_valid = label_context[self.label_context_loc].flatten()
      label_context[self.label_context_loc] = 2 * self.args.mpnet_term_weight * torch.sigmoid(label_context_valid)
    elif self.args.mpnet_smoothness_train == 'softmax':
      label_context_valid = label_context[self.label_context_loc].flatten()
      label_context_max = label_context_valid.max()
      label_context_norm = nn.Softmax(dim=0)(label_context_valid)
      label_context_norm_max = label_context_norm.max()
      label_context[self.label_context_loc] = label_context_norm * label_context_max / label_context_norm_max

    if self.args.mpnet_smoothness_train in {'sigmoid', 'softmax'}:
      label_context[self.label_context_diag_loc] = self.args.mpnet_diag_value

    assert unary.size()[-3:-1] == edge_weights.size()[-2:] if (edge_weights is not None) else True

    self.args.training = self.training
    if self.args.mpnet_mrf_mode in {'ISGMR', 'SGM'}:
      if self.args.mpnet_enable_sgm_single and (self.args.mpnet_mrf_mode == 'ISGMR'):
        cost_final, cost_all = [], []

        for dir_idx in range(self.args.mpnet_n_dirs):
          self.args.sgm_single_mode = dir_idx
          edge_weight_in = edge_weights[:, :, dir_idx : dir_idx + 1] if (edge_weights is not None) else None
          cost_final_per, cost_all_per = ISGMRFunction.apply(unary,
                                                             label_context,
                                                             edge_weight_in,
                                                             self.args)
          cost_final.append(cost_final_per.permute(0, 1, 4, 2, 3).contiguous())
          cost_all.append(cost_all_per.contiguous())
      else:
        self.args.sgm_single_mode = -1
        cost_final, cost_all = ISGMRFunction.apply(unary, label_context, edge_weights, self.args)
    elif self.args.mpnet_mrf_mode == 'TRWP':
      cost_final, cost_all = TRWPFunction.apply(unary, label_context, edge_weights, self.args)
    else:
      assert False

    cost_final = cost_final.permute(0, 1, 4, 2, 3).contiguous() if not self.args.mpnet_enable_sgm_single else cost_final
    label_context = label_context.unsqueeze(0)  # Create batch

    return cost_final, label_context, cost_all, None, None
