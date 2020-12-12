import torch, sys
import torch.nn as nn
sys.path.append('..')
from utils.edge_weights import getEdgeShift
from utils.label_context import create_label_context


class MeanField(nn.Module):
  def __init__(self, args, enable_create_label_context=True, enable_seg=None, enable_symmetric=False):
    super(MeanField, self).__init__()
    self.args = args

    if enable_create_label_context:
      label_context, _, _ = create_label_context(args)  # (21,21)
      self.label_context = label_context.view(1, self.args.n_classes, self.args.n_classes, 1, 1)

  def set_label_context(self, label_context):
    self.label_context = label_context.view(1, self.args.n_classes, self.args.n_classes, 1, 1)

  def forward(self, unary, edge_weights=None):
    # unary:(batch,1,21,256,256); edge_weights:(batch,dirs,256,256)
    unary = unary.squeeze(1)  # cost
    q_i = nn.Softmax(dim=1)(-unary)
    edge_weights = edge_weights.permute(1, 0, 2, 3).unsqueeze(2).contiguous() if (edge_weights is not None) else 1
    all_cost_iter = [] if self.args.enable_saving_label else None

    for iter in range(self.args.mpnet_max_iter):
      q_j = getEdgeShift('shift', q_i, self.args.mpnet_n_dirs)  # (dirs,batch,21,256,256)
      # TODO:mean is better than sum, also proved in energy minimization
      q_i_tild = torch.mean(edge_weights * q_j, dim=0).unsqueeze(1)  # (batch,1,21,256,256)

      if False:
        q_i_hat = torch.sum(self.label_context * q_i_tild, dim=2)
      else:
        q_i_list = []
        for l in range(self.args.n_classes):
          value = torch.sum(self.label_context[:, l] * q_i_tild[:, 0], dim=1)
          q_i_list.append(value)
        q_i_hat = torch.stack(q_i_list, dim=1)

      q_i = -unary - q_i_hat  # (batch,21,256,256), this is score

      if self.args.enable_saving_label:
        all_cost_iter.append(-q_i.unsqueeze(1))

      if iter < self.args.mpnet_max_iter - 1:  # last one left for softmax in loss
        q_i = nn.Softmax(dim=1)(q_i)

    final_cost = -q_i.unsqueeze(1)

    return [final_cost, self.label_context, all_cost_iter]  # cost
