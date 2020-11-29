import sys
import torch
import torch.nn as nn
import time
sys.path.append('../tools/python')
sys.path.append('tools/python')
from utils import check_valid, check_data_input


class Params():
  def __init__(self, n_dirs=4, mode='TRWP', graph_model='min_sum', enable_debug=False):
    self.max_iter = 5
    self.batch = 0
    self.height = 0
    self.width = 0
    self.enable_cuda = True
    self.data_type = torch.float32
    self.n_labels = 21
    self.n_dirs = n_dirs
    self.epsilon = 0
    self.device = None
    self.dir_list = [i for i in range(n_dirs)]
    self.mode = mode
    self.enable_parallel = False
    self.enable_debug = False
    self.target = 'Semantic'
    enable_loopy = True

    if mode == 'TRWP':
      enable_loopy = enable_debug
      self.enable_parallel = False

    # gamma is appearance probability
    if enable_loopy:
      self.gamma_pnode = 1
      self.gamma_pedge = 1
      self.gamma_mnode = 1
      self.gamma_medge = 1
    elif graph_model in {'max_product', 'sum_product'}:
      # standard, gamma_node=1, gamma_edge=1/(n_dirs//2), message (0,1] will be too large, overflow
      # modified version to enlarge cost, original is gammaNode=2, gammaEdge=1
      self.gamma_pnode = 1
      self.gamma_pedge = 1 / (n_dirs // 2)
      self.gamma_mnode = n_dirs // 2
      self.gamma_medge = 1
    elif graph_model == 'min_sum':
      # standard, gamma_node=1, gamma_edge=1/(n_dirs//2), set small (unary cost + message)/gamma to avoid overflow
      # TODO
      self.gamma_pnode = 2
      self.gamma_pedge = 1
      self.gamma_mnode = 2
      self.gamma_medge = 1

      # self.gamma_pnode = n_dirs // 2
      # self.gamma_pedge = 1
      # self.gamma_mnode = n_dirs // 2
      # self.gamma_medge = 1

      # self.gamma_pnode = 1
      # self.gamma_pedge = 1 / (n_dirs // 2)
      # self.gamma_mnode = n_dirs // 2
      # self.gamma_medge = 1


class MessagePassingFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, phi_i, phi_ij, message_store, params):
    time_start = time.time()
    batch = phi_i.size()[0]
    ctx.intermediate_results = phi_i, phi_ij, message_store, params

    # This is much faster than create zeros/empty tensor
    message_final = message_store[:, 0, :, :, :, :] * 0 + 1

    print('forward time inside: {:.4f}s'.format(time.time() - time_start))
    return message_final

  @staticmethod
  def backward(ctx, dmessage_final):
    time_start = time.time()
    dmessage_final = dmessage_final.contiguous()  # This is very important!

    phi_i, phi_ij, message_store, params = ctx.intermediate_results

    # This is much faster than create zeros tensor
    dphi_i = phi_i * 0
    dphi_ij = phi_ij * 0

    del ctx.intermediate_results

    print('backward time inside: {:.4f}s'.format(time.time() - time_start))
    return dphi_i, dphi_ij, None, None


class MessagePassingModule(nn.Module):
  def __init__(self, max_iter=5, n_labels=21, n_dirs=4, mode='TRWP',
               ISGMR_parallel=True, enable_debug=False, target='Semantic',
               graph_model='sum_product', label_context=None, n_edge_feats=1,
               llambda=1., enable_Gaussian=False, enable_cuda=False,
               enable_soft_weight=False, enable_message_vector=False,
               enable_loopy=False):
    super(MessagePassingModule, self).__init__()
    assert(graph_model in ['sum_product', 'max_product', 'min_sum'])
    self.reset_parameters()
    self.params = Params(n_dirs=n_dirs, mode=mode, graph_model=graph_model,
                         enable_debug=enable_loopy)
    self.params.max_iter = max_iter
    self.params.n_labels = n_labels
    self.params.n_dirs = n_dirs
    self.params.mode = mode
    self.params.enable_parallel = ISGMR_parallel
    self.params.enable_debug = enable_debug
    self.params.target = target
    self.enable_max_product = graph_model == 'max_product'
    self.enable_sum_product = graph_model == 'sum_product'
    self.enable_min_sum = graph_model == 'min_sum'
    self.enable_prob_model = self.enable_max_product or self.enable_sum_product
    self.graph_model = graph_model
    self.n_edge_feats = n_edge_feats
    self.sigma = 15
    self.llambda = llambda
    self.enable_Gaussian = enable_Gaussian
    self.enable_check_value = False
    self.enable_soft_weight = enable_soft_weight
    self.enable_message_vector = enable_message_vector

    # label_context:(n_edge_feats,n_labels,n_labels)
    if label_context is None:
      dtype = torch.float64 if self.enable_prob_model else torch.float32
      label_context = 1 - torch.eye(n_labels, dtype=dtype)
      label_context = label_context.view(1, n_labels, n_labels).repeat(n_dirs, 1, 1)

      if enable_cuda:
        label_context = label_context.cuda()

    self.label_context = nn.Parameter(label_context, requires_grad=True)  # this will affect weight_regularization
    # self.label_context = label_context  # this above is_cuda is False when switch on CUDA, TODO

  def reset_parameters(self):
    for weight in self.parameters():
      weight.data.fill_(0)

  def get_mode_params(self, dir_loc):
    h_start, h_stop, w_start, w_stop = 0, self.params.height - 1, 0, self.params.width - 1
    h_step, w_step = 0, 0
    dir_loc_inv = (dir_loc + 1) if (dir_loc % 2 == 0) else (dir_loc - 1)

    if dir_loc == 0:
      w_start, w_stop, w_step = 0, self.params.width - 2, 1
    elif dir_loc == 1:
      w_start, w_stop, w_step = self.params.width - 1, 1, -1
    elif dir_loc == 2:
      h_start, h_stop, h_step = 0, self.params.height - 2, 1
    elif dir_loc == 3:
      h_start, h_stop, h_step = self.params.height - 1, 1, -1
    elif dir_loc == 4:
      w_start, w_stop = 0, self.params.width - 2
      h_start, h_stop = 0, self.params.height - 2
      h_step, w_step = 1, 1
    elif dir_loc == 5:
      w_start, w_stop = 1, self.params.width - 1
      h_start, h_stop = self.params.height - 1, 1
      h_step, w_step = -1, -1
    elif dir_loc == 6:
      w_start, w_stop = 1, self.params.width - 1
      h_start, h_stop = 0, self.params.height - 2
      h_step, w_step = 1, -1
    elif dir_loc == 7:
      w_start, w_stop = 0, self.params.width - 2
      h_start, h_stop = self.params.height - 1, 1
      h_step, w_step = -1, 1
    elif dir_loc == 8:
      w_start, w_stop = 0, self.params.width - 3
      h_start, h_stop = 0, self.params.height - 2
      h_step, w_step = 1, 2
    elif dir_loc == 9:
      w_start, w_stop = 2, self.params.width - 1
      h_start, h_stop = self.params.height - 1, 1
      h_step, w_step = -1, -2
    elif dir_loc == 10:
      w_start, w_stop = 0, self.params.width - 2
      h_start, h_stop = 0, self.params.height - 3
      h_step, w_step = 2, 1
    elif dir_loc == 11:
      w_start, w_stop = 1, self.params.width - 1
      h_start, h_stop = self.params.height - 1, 2
      h_step, w_step = -2, -1
    elif dir_loc == 12:
      w_start, w_stop = 1, self.params.width - 1
      h_start, h_stop = 0, self.params.height - 3
      h_step, w_step = 2, -1
    elif dir_loc == 13:
      w_start, w_stop = 0, self.params.width - 2
      h_start, h_stop = self.params.height - 1, 2
      h_step, w_step = -2, 1
    elif dir_loc == 14:
      w_start, w_stop = 2, self.params.width - 1
      h_start, h_stop = 0, self.params.height - 2
      h_step, w_step = 1, -2
    elif dir_loc == 15:
      w_start, w_stop = 0, self.params.width - 3
      h_start, h_stop = self.params.height - 1, 1
      h_step, w_step = -1, 2
    else:
      assert False, 'Direction mode out of range (0~15)'

    return h_start, h_stop, h_step, w_start, w_stop, w_step, dir_loc_inv

  def update_unary_belief(self, phi_i, message):
    if self.enable_prob_model:
      message_reweight = message.pow(1 / self.params.gamma_mnode)
      unary_belief = phi_i * message_reweight.prod(dim=1, keepdim=False)
    else:
      # message_reweight = message * (1 / self.params.gamma_mnode)
      # unary_belief = phi_i + message_reweight.sum(dim=1, keepdim=False)
      unary_belief = phi_i * self.params.gamma_mnode + message.sum(dim=1, keepdim=False)

    return unary_belief

  def update_pairwise_belief(self, weight_kernel, unary_belief, message, label_context, phi_ij=None):
    if self.enable_prob_model:
      message_reweight = message.pow(1 / self.params.gamma_medge)
      pairwise_belief = unary_belief.new_ones(self.params.batch, 2, self.params.height,
                                              self.params.width, self.params.n_labels, self.params.n_labels)
      pairwise_belief += self.params.epsilon
    else:
      message_reweight = message * (1 / self.params.gamma_medge)
      pairwise_belief = unary_belief.new_zeros(self.params.batch, 2, self.params.height,
                                               self.params.width, self.params.n_labels, self.params.n_labels)

    unary_belief_left = unary_belief.unsqueeze(4).repeat(1, 1, 1, 1, self.params.n_labels)
    unary_belief_right = unary_belief.unsqueeze(3).repeat(1, 1, 1, self.params.n_labels, 1)
    message_left = message_reweight[:, 0, :, :, :].unsqueeze(3).repeat(1, 1, 1, self.params.n_labels, 1)
    message_right = message_reweight[:, 1, :, :, :].unsqueeze(4).repeat(1, 1, 1, 1, self.params.n_labels)
    message_top = message_reweight[:, 2, :, :, :].unsqueeze(3).repeat(1, 1, 1, self.params.n_labels, 1)
    message_bottom = message_reweight[:, 3, :, :, :].unsqueeze(4).repeat(1, 1, 1, 1, self.params.n_labels)

    if self.enable_prob_model:
      # Vertical
      phi_ij_tmp = self.cal_pairwise_terms_update(label_context, weight_kernel[:, :, 1, 0:-1, :, :, :]).exp() if phi_ij is None else phi_ij[:, 1, 0:-1, :, :, :]
      pairwise_belief[:, 1, 0:-1, :, :, :] = phi_ij_tmp * unary_belief_left[:, 0:-1, :, :, :] * unary_belief_right[:, 1:, :, :, :] / \
                                             message_bottom[:, 0:-1, :, :, :] / message_top[:, 1:, :, :, :]

      # Horizontal
      phi_ij_tmp = self.cal_pairwise_terms_update(label_context, weight_kernel[:, :, 0, :, 0:-1, :, :]).exp() if phi_ij is None else phi_ij[:, 0, :, 0:-1, :, :]
      pairwise_belief[:, 0, :, 0:-1, :, :] = phi_ij_tmp * unary_belief_left[:, :, 0:-1, :, :] * unary_belief_right[:, :, 1:, :, :] / \
                                             message_right[:, :, 0:-1, :, :] / message_left[:, :, 1:, :, :]
    else:
      # Vertical
      if phi_ij is None:
        phi_ij_tmp = self.cal_pairwise_terms_update(label_context, weight_kernel[:, :, 1, 0:-1, :, :, :])
      else:
        phi_ij_tmp = phi_ij[:, 1, 0:-1, :, :, :]
        if self.enable_check_value:
          assert (self.cal_pairwise_terms_update(label_context, weight_kernel[:, :, 1, 0:-1, :, :, :]) - phi_ij_tmp).abs().sum() == 0

      pairwise_belief[:, 1, 0:-1, :, :, :] = phi_ij_tmp + unary_belief_left[:, 0:-1, :, :, :] + unary_belief_right[:, 1:, :, :, :] - \
                                             message_bottom[:, 0:-1, :, :, :] - message_top[:, 1:, :, :, :]

      # Horizontal
      if phi_ij is None:
        phi_ij_tmp = self.cal_pairwise_terms_update(label_context, weight_kernel[:, :, 0, :, 0:-1, :, :])
      else:
        phi_ij_tmp = phi_ij[:, 0, :, 0:-1, :, :]
        if self.enable_check_value:
          assert (self.cal_pairwise_terms_update(label_context, weight_kernel[:, :, 0, :, 0:-1, :, :]) - phi_ij_tmp).abs().sum() == 0

      pairwise_belief[:, 0, :, 0:-1, :, :] = phi_ij_tmp + unary_belief_left[:, :, 0:-1, :, :] + unary_belief_right[:, :, 1:, :, :] - \
                                             message_right[:, :, 0:-1, :, :] - message_left[:, :, 1:, :, :]

    return pairwise_belief

  def cal_pairwise_terms_update(self, pairwise_func, pairwise_weight):
    # pairwise_weight:(batch,nfeats,h,w,1,1), pairwise_func:(1,nfeats,1,1,nlabels,nlabels)
    batch, n_edge_feats, h, w, _, _ = pairwise_weight.size()
    pairwise_cost = torch.matmul(pairwise_weight.permute(0, 2, 3, 4, 5, 1).reshape(batch, -1, n_edge_feats),
                                 pairwise_func.reshape(1, n_edge_feats, -1))
    return pairwise_cost.view(batch, h, w, self.params.n_labels, self.params.n_labels)

  def cal_pairwise_terms(self, pairwise_func, pairwise_weight):
    # pairwise_weight:(batch,nfeats,h/w,1,1), pairwise_func:(1,nfeats,1,nlabels,nlabels)
    # pairwise_cost = torch.matmul(pairwise_weight.permute(0, 2, 3, 4, 1).view(self.params.batch, -1, self.n_edge_feats),
    #                              pairwise_func.view(1, self.n_edge_feats, -1))
    pairwise_cost = pairwise_func * 1  # TODO
    return pairwise_cost.view(1, -1, self.params.n_labels, self.params.n_labels) \
      .repeat(self.params.batch, 1, 1, 1)

  def cal_weight_kernel(self, img):
    weight_kernel = None

    if img is not None:
      img = img.mul(255) if img.max() <= 2 else img  # img would be normalized in dataloader
      channel, n_kernels = img.size()[1], self.params.n_dirs // 2

      if self.enable_Gaussian:
        weight_kernel = img.new_zeros(self.params.batch, self.n_edge_feats, self.params.n_dirs // 2, self.params.height, self.params.width)

        for dir_loc in range(0, self.params.n_dirs, 2):
          img_shift = img.new_zeros(self.params.batch, channel, self.params.height, self.params.width)
          [h_start, h_stop, h_step, w_start, w_stop, w_step, _] = self.get_mode_params(dir_loc=dir_loc)
          img_shift[:, :, h_start:h_stop + 1, w_start:w_stop + 1] = \
            img[:, :, (h_start + h_step):(h_stop + 1 + h_step), (w_start + w_step):(w_stop + 1 + w_step)]
          img_diff = (img_shift - img).div(self.sigma).pow(2).sum(dim=1).div(channel).neg().exp()
          euclidean_dis = math.sqrt(h_step**2 + w_step**2)
          weight_kernel[:, 0, dir_loc // 2, h_start:h_stop + 1, w_start:w_stop + 1] = \
            img_diff[:, h_start:h_stop + 1, w_start:w_stop + 1] / euclidean_dis
          assert check_valid(weight_kernel), 'Invalid kernel value(s)'

          # # Test this in small_test.py
          # if self.params.enable_debug:
          #   print('Create weight kernel, dirction:', dir_loc)
          #   weight_kernel_current = weight_kernel[:, dir_loc // 2, :, :]
          #   check_data_input([img.long(), img_shift.long(),
          #                     weight_kernel_current.mul(255/weight_kernel_current.max()).long()], time=1)
      else:
        weight_kernel = img.new_ones(self.params.batch, self.n_edge_feats, self.params.n_dirs // 2, self.params.height, self.params.width)

      if self.params.enable_debug:
        print('Check weight kernel, min: {:.4f}, max: {:.4f}, mean: {:.4f}' \
              .format(weight_kernel.min(), weight_kernel.max(), weight_kernel.mean()))

    return weight_kernel * 0  # TODO

  def cross(self, phi_i, message, weight_kernel, label_context, edge_weights, dir_loc, phi_ij=None,
            message_vector=None, message_index=None):
    # phi_i:(batch,height,width,n_labels)
    # phi_ij:(batch,2,height,width,n_labels,n_labels)
    # message:(batch,n_dirs,height,width,n_labels)
    # node_vector:(batch,height/width,n_labels)
    # edge_vector:(batch,height/width,n_labels,n_labels)
    # edge_weights:(batch,height,width)
    # message_vector:(batch,height,width,n_labels,n_labels)
    # message_indx:(batch,height,width)
    message_out = message.clone()
    [h_start, h_stop, h_step, w_start, w_stop, w_step, dir_loc_inv] = self.get_mode_params(dir_loc=dir_loc)

    if self.enable_prob_model:
      new_message_a_dir = message.new_ones(self.params.batch, self.params.height, self.params.width, self.params.n_labels)
    else:
      new_message_a_dir = message.new_zeros(self.params.batch, self.params.height, self.params.width, self.params.n_labels)

    dir_list = list(range(self.params.n_dirs))
    dir_list.remove(dir_loc)
    dir_mode_region = dir_loc // 2
    is_forward = dir_loc % 2 == 0

    # remove one of height or weight
    label_context = label_context.view(1, self.n_edge_feats, 1, self.params.n_labels, self.params.n_labels)
    edge_weights = edge_weights.view(self.params.batch, self.params.height, self.params.width, 1, 1) \
      .repeat(1, 1, 1, self.params.n_labels, self.params.n_labels)

    if dir_mode_region == 0:  # horizontal
      for w in range(w_start, w_stop + w_step, w_step):
        message_conc = message[:, dir_list, :, w, :]
        msg_temp = torch.cat((new_message_a_dir[:, :, w, :].unsqueeze(1), message_conc), dim=1)

        if self.enable_prob_model:
          msg_temp = msg_temp.pow(1 / self.params.gamma_mnode)
          node_vector = phi_i[:, :, w, :] * msg_temp.prod(dim=1)
          message_inv = message[:, dir_loc_inv, :, w, :].pow(1 / self.params.gamma_medge)
        else:
          msg_temp = msg_temp * (1 / self.params.gamma_mnode)
          node_vector = phi_i[:, :, w, :] + msg_temp.sum(dim=1)
          # TODO
          # node_vector_min, _ = torch.min(node_vector, dim=2, keepdim=True)
          # node_vector = node_vector - node_vector_min
          message_inv = message[:, dir_loc_inv, :, w, :] * (1 / self.params.gamma_medge)

        if is_forward:
          node_vector = node_vector.unsqueeze(3).repeat(1, 1, 1, self.params.n_labels)
          message_inv = message_inv.unsqueeze(3).repeat(1, 1, 1, self.params.n_labels)

          if phi_ij is None:
            phi_ij_tmp = self.cal_pairwise_terms(label_context, weight_kernel[:, :, dir_mode_region, :, w, :, :])
          else:
            phi_ij_tmp = phi_ij[:, dir_mode_region, :, w, :, :]
            if self.enable_check_value:
              pairwise_cost = self.cal_pairwise_terms(label_context, weight_kernel[:, :, dir_mode_region, :, w, :, :])
              pairwise_cost = pairwise_cost.exp() if self.enable_prob_model else pairwise_cost
              assert (pairwise_cost - phi_ij_tmp).abs().sum() == 0

          phi_ij_tmp = phi_ij_tmp.repeat(1,self.params.height,1,1) * edge_weights[:,:,w+w_step,:,:]
          edge_collect_dim = 2
        else:
          node_vector = node_vector.unsqueeze(2).repeat(1, 1, self.params.n_labels, 1)
          message_inv = message_inv.unsqueeze(2).repeat(1, 1, self.params.n_labels, 1)

          if phi_ij is None:
            phi_ij_tmp = self.cal_pairwise_terms(label_context, weight_kernel[:, :, dir_mode_region, :, w + w_step, :, :])
          else:
            phi_ij_tmp = phi_ij[:, dir_mode_region, :, w + w_step, :, :]
            if self.enable_check_value:
              pairwise_cost = self.cal_pairwise_terms(label_context, weight_kernel[:, :, dir_mode_region, :, w + w_step, :, :])
              pairwise_cost = pairwise_cost.exp() if self.enable_prob_model else pairwise_cost
              assert (pairwise_cost - phi_ij_tmp).abs().sum() == 0

          phi_ij_tmp = phi_ij_tmp.repeat(1,self.params.height,1,1) * edge_weights[:, :, w + w_step, :, :]
          edge_collect_dim = 3

        if self.enable_prob_model:
          phi_ij_tmp = phi_ij_tmp.exp() if phi_ij is None else phi_ij_tmp
          edge_vector = node_vector * phi_ij_tmp / message_inv

          if self.enable_max_product:
            new_message, _ = edge_vector.max(dim=edge_collect_dim)
          else:
            new_message = edge_vector.sum(dim=edge_collect_dim)

          new_message_norm = new_message / new_message.sum(dim=2, keepdim=True)
        else:
          edge_vector = node_vector + phi_ij_tmp - message_inv

          if self.enable_soft_weight:
            new_message_prob = nn.Softmax(dim=edge_collect_dim)(-edge_vector)
            new_message = (new_message_prob * edge_vector).sum(edge_collect_dim)
            # TODO
            # if w + w_step == 47:
            #   print(new_message[0, 19], new_message[0, 19].argmin())
          else:
            new_message, _ = edge_vector.min(dim=edge_collect_dim)

          new_message_min, norm_index = torch.min(new_message, dim=2, keepdim=True)
          new_message_norm = new_message - new_message_min

        new_message_a_dir[:, :, w + w_step, :] = new_message_norm

        if message_vector is not None:
          message_vector[:, :, w + w_step, :, :] = edge_vector

        if message_index is not None:
          message_index[:, :, w + w_step] = norm_index.squeeze(2)
    elif dir_mode_region == 1:  # vertical
      for h in range(h_start, h_stop + h_step, h_step):
        message_conc = message[:, dir_list, h, :, :]
        msg_temp = torch.cat((new_message_a_dir[:, h, :, :].unsqueeze(1), message_conc), dim=1)

        if self.enable_prob_model:
          msg_temp = msg_temp.pow(1 / self.params.gamma_mnode)
          node_vector = phi_i[:, h, :, :] * msg_temp.prod(dim=1)
          message_inv = message[:, dir_loc_inv, h, :, :].pow(1 / self.params.gamma_medge)
        else:
          msg_temp = msg_temp * (1 / self.params.gamma_mnode)
          node_vector = phi_i[:, h, :, :] + msg_temp.sum(dim=1)
          message_inv = message[:, dir_loc_inv, h, :, :] * (1 / self.params.gamma_medge)
          # node_vector_min, _ = torch.min(node_vector, dim=2, keepdim=True)
          # node_vector = node_vector - node_vector_min

        if is_forward:
          node_vector = node_vector.unsqueeze(3).repeat(1, 1, 1, self.params.n_labels)
          message_inv = message_inv.unsqueeze(3).repeat(1, 1, 1, self.params.n_labels)

          if phi_ij is None:
            phi_ij_tmp = self.cal_pairwise_terms(label_context, weight_kernel[:, :, dir_mode_region, h, :, :, :])
          else:
            phi_ij_tmp = phi_ij[:, dir_mode_region, h, :, :, :]
            if self.enable_check_value:
              pairwise_cost = self.cal_pairwise_terms(label_context, weight_kernel[:, :, dir_mode_region, h, :, :, :])
              pairwise_cost = pairwise_cost.exp() if self.enable_prob_model else pairwise_cost
              assert (pairwise_cost - phi_ij_tmp).abs().sum() == 0

          phi_ij_tmp = phi_ij_tmp.repeat(1,self.params.width,1,1) * edge_weights[:, h+h_step, :, :, :]
          edge_collect_dim = 2
        else:
          node_vector = node_vector.unsqueeze(2).repeat(1, 1, self.params.n_labels, 1)
          message_inv = message_inv.unsqueeze(2).repeat(1, 1, self.params.n_labels, 1)

          if phi_ij is None:
            phi_ij_tmp = self.cal_pairwise_terms(label_context, weight_kernel[:, :, dir_mode_region, h + h_step, :, :, :])
          else:
            phi_ij_tmp = phi_ij[:, dir_mode_region, h + h_step, :, :, :]
            if self.enable_check_value:
              pairwise_cost = self.cal_pairwise_terms(label_context, weight_kernel[:, :, dir_mode_region, h + h_step, :, :, :])
              pairwise_cost = pairwise_cost.exp() if self.enable_prob_model else pairwise_cost
              assert (pairwise_cost - phi_ij_tmp).abs().sum() == 0

          phi_ij_tmp = phi_ij_tmp.repeat(1,self.params.width,1,1) * edge_weights[:, h + h_step, :, :, :]
          edge_collect_dim = 3

        if self.enable_prob_model:
          phi_ij_tmp = phi_ij_tmp.exp() if phi_ij is None else phi_ij_tmp
          edge_vector = node_vector * phi_ij_tmp / message_inv

          if self.enable_max_product:
            new_message, _ = edge_vector.max(dim=edge_collect_dim)
          else:
            new_message = edge_vector.sum(dim=edge_collect_dim)

          new_message_norm = new_message / new_message.sum(dim=2, keepdim=True)
        else:
          edge_vector = node_vector + phi_ij_tmp - message_inv

          if self.enable_soft_weight:
            new_message_prob = nn.Softmax(dim=edge_collect_dim)(-edge_vector)
            new_message = (new_message_prob * edge_vector).sum(edge_collect_dim)
          else:
            new_message, _ = edge_vector.min(dim=edge_collect_dim)

          new_message_min, norm_index = torch.min(new_message, dim=2, keepdim=True)
          new_message_norm = new_message - new_message_min

        new_message_a_dir[:, h + h_step, :, :] = new_message_norm

        if message_vector is not None:
          message_vector[:, h + h_step, :, :, :] = edge_vector

        if message_index is not None:
          message_index[:, h + h_step, :] = norm_index.squeeze(2)
    else:  # all other directions
      w_stop_include = w_stop + 1
      h_move = 1 if h_step > 0 else -1
      for h in range(h_start, h_stop + h_move, h_move):
        message_conc = message[:, dir_list, h, w_start:w_stop_include, :]
        msg_temp = torch.cat((new_message_a_dir[:, h, w_start:w_stop_include, :].unsqueeze(1), message_conc), dim=1)

        if self.enable_prob_model:
          msg_temp = msg_temp.pow(1 / self.params.gamma_mnode)
          node_vector = phi_i[:, h, w_start:w_stop_include, :] * msg_temp.prod(dim=1)
          message_inv = message[:, dir_loc_inv, h, w_start:w_stop_include, :].pow(1 / self.params.gamma_medge)
        else:
          msg_temp = msg_temp * (1 / self.params.gamma_mnode)
          node_vector = phi_i[:, h, w_start:w_stop_include, :] + msg_temp.sum(dim=1)
          message_inv = message[:, dir_loc_inv, h, w_start:w_stop_include, :] * (1 / self.params.gamma_medge)
          # node_vector_min, _ = torch.min(node_vector, dim=2, keepdim=True)
          # node_vector = node_vector - node_vector_min

        if is_forward:
          node_vector = node_vector.unsqueeze(3).repeat(1, 1, 1, self.params.n_labels)
          message_inv = message_inv.unsqueeze(3).repeat(1, 1, 1, self.params.n_labels)
          phi_ij_tmp = self.cal_pairwise_terms(label_context, weight_kernel[:, :, dir_mode_region, h, w_start:w_stop_include, :, :])
          edge_collect_dim = 2
          phi_ij_tmp = phi_ij_tmp.repeat(1, w_stop_include - w_start, 1, 1) * edge_weights[:, h + h_step, (w_start + w_step):(w_stop_include + w_step), :, :]
        else:
          node_vector = node_vector.unsqueeze(2).repeat(1, 1, self.params.n_labels, 1)
          message_inv = message_inv.unsqueeze(2).repeat(1, 1, self.params.n_labels, 1)
          phi_ij_tmp = self.cal_pairwise_terms(label_context , weight_kernel[:, :, dir_mode_region, h + h_step, (w_start + w_step):(w_stop_include + w_step), :, :])
          edge_collect_dim = 3
          phi_ij_tmp = phi_ij_tmp.repeat(1, w_stop_include - w_start, 1, 1) * edge_weights[:, h + h_step, (w_start + w_step):(w_stop_include + w_step), :, :]

        if self.enable_prob_model:
          phi_ij_tmp = phi_ij_tmp.exp()
          edge_vector = node_vector * phi_ij_tmp / message_inv

          if self.enable_max_product:
            new_message, _ = edge_vector.max(dim=edge_collect_dim)
          else:
            new_message = edge_vector.sum(dim=edge_collect_dim)

          new_message_norm = new_message / new_message.sum(dim=2, keepdim=True)
        else:
          edge_vector = node_vector + phi_ij_tmp - message_inv

          if self.enable_soft_weight:
            new_message_prob = nn.Softmax(dim=edge_collect_dim)(-edge_vector)
            new_message = (new_message_prob * edge_vector).sum(edge_collect_dim)
          else:
            new_message, _ = edge_vector.min(dim=edge_collect_dim)

          new_message_min, norm_index = torch.min(new_message, dim=2, keepdim=True)
          new_message_norm = new_message - new_message_min

        new_message_a_dir[:, h + h_step, (w_start + w_step):(w_stop_include + w_step), :] = new_message_norm

        if message_vector is not None:
          message_vector[:, h + h_step, (w_start + w_step):(w_stop_include + w_step), :, :] = edge_vector

        if message_index is not None:
          message_index[:, h + h_step, (w_start + w_step):(w_stop_include + w_step)] = norm_index.squeeze(2)

    if self.params.mode == 'ISGMR' and self.params.enable_parallel:
      return new_message_a_dir
    else:
      message_out[:, dir_loc, :, :, :] = new_message_a_dir
      return message_out

  # standard model unary_score:(batch,n_labels,height,width), transposed to (batch,height,width,n_labels);
  # customized pairwise_score:(batch,n_labels,n_labels,2,height,width) to (batch,2,height,width,n_labels,n_labels)
  # customized message:(batch,n_dirs,height,width,n_labels)
  def forward(self, unary_score, img, edge_weights=None, pairwise_feat=None, pairwise_score=None):
    batch, n_labels, height, width = unary_score.size()

    self.params.height = height
    self.params.width = width
    self.params.batch = batch
    self.params.n_labels = n_labels
    self.params.enable_cuda = unary_score.is_cuda
    self.params.data_type = unary_score.dtype
    self.params.device = unary_score.device

    unary_score = unary_score.permute(0, 2, 3, 1).contiguous()
    unary_score_reweight = unary_score * (1 / self.params.gamma_pnode)

    label_context = self.label_context.view(1, self.n_edge_feats, self.params.n_dirs, 1, 1, n_labels, n_labels)

    if edge_weights is None:
      edge_weights = unary_score.new_ones(self.params.batch, self.params.n_dirs, self.params.height, self.params.width)

    if (pairwise_score is not None) and (pairwise_feat is not None):
      pairwise_score = pairwise_score.permute(0, 3, 4, 5, 1, 2).contiguous()
      pairwise_score_reweight = pairwise_score * (1 / self.params.gamma_pedge)
      # pairwise_feat stands for score
      weight_kernel = pairwise_feat.view(batch, self.n_edge_feats, self.params.n_dirs // 2, height, width, 1, 1)
      self.enable_check_value = True
    else:
      pairwise_score, pairwise_score_reweight = None, None
      # Get weight kernel (Gaussian), cal_weight_kernel stands for cost
      weight_kernel = self.cal_weight_kernel(img).view(batch, self.n_edge_feats, self.params.n_dirs // 2, height, width, 1, 1)
      weight_kernel = -weight_kernel  # make to score type
      self.enable_check_value = False

    weight_kernel_reweighted = weight_kernel * (1 / self.params.gamma_pedge) * self.llambda

    # if input is unary_cost, then unary_score=-unary_cost
    if self.enable_prob_model:
      phi_i = unary_score_reweight.exp()
      phi_ij = pairwise_score_reweight.exp() if pairwise_score_reweight is not None else None
      message = torch.ones(batch, self.params.n_dirs, height, width, n_labels,
                           device=self.params.device, dtype=self.params.data_type)
    else:
      phi_i = -unary_score_reweight
      phi_ij = -pairwise_score_reweight if pairwise_score_reweight is not None else None
      message_init = torch.zeros(batch, self.params.n_dirs, height, width, n_labels,
                                 device=self.params.device, dtype=self.params.data_type, requires_grad=True)
      message = message_init
      weight_kernel_reweighted = -weight_kernel_reweighted  # changed to cost type for energy

    if self.params.enable_debug and self.params.target == 'Stereo':
      segs = []

    duration_sum = 0

    if self.enable_message_vector:
      message_vector = torch.zeros(self.params.max_iter, batch, self.params.n_dirs, height, width, n_labels, n_labels,
                                   device=self.params.device, dtype=self.params.data_type)
      message_index = torch.zeros(self.params.max_iter, batch, self.params.n_dirs, height, width,
                                  device=self.params.device, dtype=torch.uint8)
    else:
      message_vector = None
      message_index = None

    for iter in range(self.params.max_iter):
      time_start = time.time()
      if self.params.mode == 'ISGMR' and self.params.enable_parallel:
        message_new = message.clone()
        for dir_loc in range(self.params.n_dirs):
          message_vector_one_iter = message_vector[iter, :, dir_loc] if message_vector is not None else None
          message_index_one_iter = message_index[iter, :, dir_loc] if message_index is not None else None
          message_new[:, dir_loc, :, :, :] = \
            self.cross(phi_i, message, weight_kernel_reweighted, label_context[:,:,dir_loc,:,:,:,:],
                       edge_weights[:,dir_loc,:,:], dir_loc, phi_ij=phi_ij,
                       message_vector=message_vector_one_iter,
                       message_index=message_index_one_iter)
        message = message_new
      else:
        for dir_loc in range(self.params.n_dirs):
          message_vector_one_iter = message_vector[iter, :, dir_loc] if message_vector is not None else None
          message_index_one_iter = message_index[iter, :, dir_loc] if message_index is not None else None
          # if (dir_loc <= 3): continue
          message = self.cross(phi_i, message, weight_kernel_reweighted, label_context[:,:,dir_loc,:,:,:,:],
                               edge_weights[:,dir_loc,:,:], dir_loc, phi_ij=phi_ij,
                               message_vector=message_vector_one_iter,
                               message_index=message_index_one_iter)

      duration = time.time() - time_start
      duration_sum += duration

      if self.params.enable_debug and self.enable_prob_model and not check_valid(message, enable_zero=True):
        print('Extreme check, message, min: {:.4f}, max: {:.4f}'
              .format(message.min(), message.max()))
        print('Extreme check, unary_score, min: {:.4f}, max: {:.4f}, pairwise_score, min: {:.4f}, max: {:.4f}'
              .format(unary_score.min(), unary_score.max(), pairwise_score.min(), pairwise_score.max()))

        if phi_ij is not None:
          print('Extreme check, phi_i, min: {:.4f}, max: {:.4f}, phi_ij, min: {:.4f}, max: {:.4f}'
                .format(phi_i.min(), phi_i.max(), phi_ij.min(), phi_ij.max()))
        assert False

      if self.params.enable_debug and (self.params.max_iter > 1) and (self.params.target == 'Stereo') \
              or ((self.params.max_iter >= 100) and (iter % 2 == 0)):
        print('Iter: {}, model: {}-{}, h: {}, w: {}, dirs: {}, labels: {}, time: {:.4f}s' \
              .format(iter, self.params.mode, self.graph_model, self.params.height, self.params.width,
                      self.params.n_dirs, self.params.n_labels, duration))
        unary_belief_debug = self.update_unary_belief(phi_i, message)
        seg = unary_belief_debug.argmax(dim=3) if self.enable_prob_model else unary_belief_debug.argmin(dim=3)
        segs.append(seg)
        if self.params.max_iter > 20:
          check_data_input(seg, time=1)

    if self.params.enable_debug and (self.params.max_iter > 1) and (self.params.target == 'Stereo'):
      check_data_input(segs, time=None)
      segs = np.stack(segs, axis=3)

      scio.savemat('../result/energy/tsukuba/{}_{}_dir{}_Gau{}_iter{}.mat' \
                   .format(self.params.mode, self.graph_model.replace('_', '-'), self.params.n_dirs,
                           1 if self.enable_Gaussian else 0, self.params.max_iter),
                   {'segs': segs, 'time': duration_sum / self.params.max_iter})

    unary_belief = self.update_unary_belief(phi_i, message)

    # TODO pairwise_belief is too expensive
    # pairwise_belief = self.update_pairwise_belief(weight_kernel_reweighted, unary_belief, message, label_context, phi_ij=phi_ij)

    if self.enable_prob_model:
      unary_prob = unary_belief / unary_belief.sum(dim=3, keepdim=True)
      # pairwise_prob = pairwise_belief / pairwise_belief.sum(dim=(4, 5), keepdim=True)
    else:
      unary_prob = -unary_belief
      # unary_prob = nn.Softmax(dim=3)(-unary_belief)
      # pairwise_prob = nn.Softmax(dim=4)(-pairwise_belief.view(batch, 2, height, width, -1))
      # pairwise_prob = pairwise_prob.view(batch, 2, height, width, n_labels, n_labels)

    unary_prob = unary_prob.permute(0, 3, 1, 2).contiguous()
    # pairwise_prob = pairwise_prob.permute(0, 4, 5, 1, 2, 3).contiguous()
    pairwise_prob = None

    # (n_iter,batch,n_dir,h,w,d,d)->(n_iter,n_dir,batch,cv,h,w,d,d)
    message_vector = message_vector.permute(0, 2, 1, 3, 4, 5, 6).unsqueeze(3)

    #(n_iter,batch,n_dir,h,w)->(n_iter,n_dir,batch,cv,h,w)
    message_index = message_index.permute(0, 2, 1, 3, 4).unsqueeze(3)

    return unary_prob, pairwise_prob, message, message_init, label_context, message_vector, message_index
