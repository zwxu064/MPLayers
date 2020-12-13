import torch, sys
import torch.nn as nn
sys.path.append('..')
from MPLayers.lib_stereo import TRWP_hard_soft as TRWP_stereo
from MPLayers.lib_seg import TRWP_hard_soft as TRWP_seg
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

    if edge_weights is None:
      edge_weights = unary.new_ones(n_dir, batch, cv, h, w)
      enable_edge_weights = False
    else:
      enable_edge_weights = True

    if args.enable_saving_label:
      label_all = unary.new_zeros(n_iter, batch, cv, h, w, dtype=torch.uint8)
    else:
      label_all = torch.empty(0, dtype=torch.uint8)

    if args.mpnet_enable_soft:
      if is_training:
        message_edge_label = unary.new_zeros(n_iter, n_dir, batch, cv, h, w, n_disp, n_disp)
        cost_index = unary.new_zeros(n_iter, n_dir, batch, cv, h, w, dtype=torch.uint8)
      else:
        message_edge_label = torch.empty(0, dtype=torch.float32)
        cost_index = torch.empty(0, dtype=torch.uint8)

      TRWP.forward_soft(rho, int(n_iter), unary, label_context, edge_weights,
                        message, message_edge_label, cost_index, cost_final,
                        unary_update, label_all)
      message_index = torch.empty(0, dtype=torch.float32)
    else:
      if is_training:
        message_index = unary.new_zeros(n_iter, n_dir, batch, cv, h, w, n_disp, dtype=torch.uint8)
        cost_index = unary.new_zeros(n_iter, n_dir, batch, cv, h, w, dtype=torch.uint8)
      else:
        message_index = torch.empty(0, dtype=torch.uint8)
        cost_index = torch.empty(0, dtype=torch.uint8)

      TRWP.forward(rho, int(n_iter), 0, unary, label_context,
                   edge_weights, message, cost_final, message_index, cost_index,
                   unary_update, label_all)
      message_edge_label = torch.empty(0, dtype=torch.float32)

    ctx.intermediate_results = rho, args, message_edge_label, message_index, \
                               cost_index, label_context, edge_weights, enable_edge_weights
    del message, message_index, unary_update, label_context, edge_weights

    return cost_final, label_all, message_edge_label, cost_index

  @staticmethod
  def backward(ctx, dcost_final, dlabel_all, dmessage_edge_label, dcost_index):
    dcost_final = dcost_final.contiguous()

    rho, args, message_edge_label, message_index, cost_index, \
      label_context, edge_weights, enable_edge_weights = ctx.intermediate_results
    del ctx.intermediate_results

    cost_index = args.msg_norm_index if (args.msg_norm_index is not None) else cost_index

    n_iter, n_dir, batch, cv, h, w = cost_index.size()
    n_disp = args.n_classes
    TRWP = TRWP_seg if (n_disp == 21) else TRWP_stereo

    dunary = dcost_final.new_zeros(batch, cv, h, w, n_disp)
    dmessage = dcost_final.new_zeros(n_dir, batch, cv, h, w, n_disp)
    dunary_update = dcost_final.new_zeros(batch, cv, h, w, n_disp)
    dedge_weights = dcost_final.new_zeros(n_dir, batch, cv, h, w)

    if args.enable_seg:
      dlabel_context = dcost_final.new_zeros(n_disp, n_disp)
    else:
      dlabel_context = dcost_final.new_zeros(n_disp)

    if args.mpnet_enable_soft:
      TRWP.backward_soft(rho, dcost_final, label_context, edge_weights, message_edge_label,
                         cost_index, dunary, dlabel_context, dedge_weights,
                         dmessage, dunary_update)
    else:
      TRWP.backward(rho, label_context, edge_weights, dcost_final, message_index,
                    cost_index, dunary, dlabel_context, dedge_weights,
                    dmessage, dunary_update)

    del message_edge_label, message_index, cost_index, label_context, \
      edge_weights, dcost_final, dmessage, dunary_update

    dedge_weights = None if (not enable_edge_weights) else dedge_weights

    return dunary, dlabel_context, dedge_weights, None, None, None


class MPModule(torch.nn.Module):
  def __init__(self, args, enable_create_label_context=False, enable_saving_label=False):
    super(MPModule, self).__init__()
    self.args = args
    self.args.enable_saving_label = enable_saving_label
    self.args.rho = 0.5 if (args.mpnet_mrf_mode == 'TRWP') else 1
    self.args.enable_seg = True if (args.n_classes == 21) else False

    self.smoothness_train = args.mpnet_smoothness_train if args.mpnet_smoothness_train else None
    self.smoothness_mode = args.mpnet_smoothness_mode if args.mpnet_smoothness_mode else None
    self.smoothness_trunct_value = args.mpnet_smoothness_trunct_value
    self.smoothness_trunct_loc = args.mpnet_smoothness_trunct_loc

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

  def forward(self, unary, edge_weights=None, msg_norm_index=None, pairwise_terms=None):
    # unary:(batch,cv,n_disp,h,w); label_context:(n_disp,n_disp) for seg and (n_disp) for stereo
    # edge_weights:(batch,n_dir,h,w) unsqueeze(1) and permute to be (n_dir,batch,cv,h,w)
    unary = unary.permute(0, 1, 3, 4, 2).contiguous()

    if True:
      edge_weights = edge_weights.unsqueeze(1).permute(2, 0, 1, 3, 4).contiguous() \
        if (edge_weights is not None) else edge_weights
    else:
      # TODO : switch on for debugging when n_cv > 1 in test_parallel_grad.py
      edge_weights = edge_weights.unsqueeze(0).permute(2, 0, 1, 3, 4).contiguous() \
        if (edge_weights is not None) else edge_weights

    label_context = self.label_context * self.args.mpnet_term_weight

    if self.args.mpnet_smoothness_train == 'sigmoid':
      label_context_valid = label_context[self.label_context_loc].flatten()
      label_context[self.label_context_loc] = 2 * torch.sigmoid(label_context_valid)
    elif self.args.mpnet_smoothness_train == 'softmax':
      label_context_valid = label_context[self.label_context_loc].flatten()
      label_context_max = label_context_valid.max()
      label_context_norm = nn.Softmax(dim=0)(label_context_valid)
      label_context_norm_max = label_context_norm.max()
      label_context[self.label_context_loc] = label_context_norm * label_context_max / label_context_norm_max

    if self.args.mpnet_smoothness_train in {'sigmoid', 'softmax'}:
      label_context[self.label_context_diag_loc] = self.args.mpnet_diag_value

    if edge_weights is not None:
      assert unary.size()[-3:-1] == edge_weights.size()[-2:]

    if unary.is_cuda and (msg_norm_index is not None):
      msg_norm_index = msg_norm_index.cuda()
    self.args.msg_norm_index = msg_norm_index

    self.args.training = self.training
    if self.args.mpnet_mrf_mode == 'TRWP':
      cost_final, cost_all, message_vector, message_index = \
        TRWPFunction.apply(unary, label_context, edge_weights, self.args)
    else:
      assert False

    cost_final = cost_final.permute(0, 1, 4, 2, 3).contiguous()
    label_context = label_context.unsqueeze(0)  # Create a batch
    return cost_final, label_context, cost_all, message_vector, message_index
