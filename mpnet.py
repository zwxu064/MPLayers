import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.deeplab import DeepLab
from pytorch.MP_module_hard_soft import MPModule as MPModule_TRWP
from pytorch.MP_module import MPModule as MPModule_ISGMR
from utils.model_init import init_params
from utils.edge_weights import getEdgeShift
from mean_field import MeanField


class MPNet(nn.Module):
  def __init__(self, args):
    super(MPNet, self).__init__()
    self.enable_mp_layer = (args.mpnet_mrf_mode in ['TRWP', 'ISGMR', 'MeanField', 'SGM'])
    self.args = args
    BatchNorm = SynchronizedBatchNorm2d if args.sync_bn else nn.BatchNorm2d
    self.enable_score_scale = args.enable_score_scale

    self.deeplab = DeepLab(num_classes=args.n_classes,
                           backbone=args.deeplab_backbone,
                           output_stride=args.deeplab_outstride,
                           sync_bn=args.deeplab_sync_bn,
                           freeze_bn=args.deeplab_freeze_bn,
                           enable_interpolation=args.deeplab_enable_interpolation,
                           pretrained_path=args.resnet_pretrained_path,
                           norm_layer=BatchNorm,
                           enable_aspp=not self.args.disable_aspp)

    if self.enable_mp_layer:
      if self.args.mpnet_mrf_mode == 'TRWP':
        self.mp_layer = MPModule_TRWP(self.args, enable_create_label_context=True, enable_saving_label=False)
      elif self.args.mpnet_mrf_mode in {'ISGMR', 'SGM'}:
        self.mp_layer = MPModule_ISGMR(self.args, enable_create_label_context=True, enable_saving_label=False)
      elif self.args.mpnet_mrf_mode == 'MeanField':
        self.mp_layer = MeanField(self.args, enable_create_label_context=True)
      else:
        assert False

  def set_enable_mplayer(self, enable_mplayer):
    self.enable_mp_layer = enable_mplayer

  # Inputs:(batch,c,h,w); MPNet, unary:(batch,cv,n_disp,h,w), edge_weights:(batch,n_dir,h,w)
  def forward(self, inputs, edge_weights=None, gt=None, image_name=None, image_size=None):
    # assert isinstance(edge_weights, list) if (edge_weights is not None) else True, \
    #   'edge_weights should be list for multi-scale'

    # !!! New 2020-01-15: feat_before_last_conv is right before the last conv of decoder, use this to create
    # pairwise terms (n_dirs,batch,h,w,n_classes,n_classes), memory-expensive, but try this first
    scores, lower_level_feats, feat_before_last_conv = self.deeplab(inputs)
    score_org = scores

    # if self.enable_score_scale:
    #   scores = scores - scores.min(1, keepdim=True)[0]
    #   score_scale = scores.max(1, keepdim=True)[0]  # scores.max(1, keepdim=True)[0] or scores.sum(dim=1, keepdim=True)
    #   scores = scores / score_scale
    #   scores = nn.LayerNorm(scores.size()[1:], eps=0, elementwise_affine=False)(scores)  # still to large value range
    #   print('====>', scores.min(), scores.max())

    input_size = inputs.size()[-2:]

    if False:
      lower_level_feat_save = lower_level_feats
      for feat_idx in range(len(lower_level_feats)):
        # num_feats in a list: 64,256,512,1024,2048
        lower_level_feat_per = F.interpolate(lower_level_feats[feat_idx],
                                             size=inputs[0].size()[-2:],
                                             mode='bilinear',
                                             align_corners=True)
        lower_level_feat_per = lower_level_feat_per[:, :, :image_size[0], :image_size[1]]
        lower_level_feat_save[feat_idx] = lower_level_feat_per.permute(0,2,3,1)

      num_batchs = lower_level_feat_save[0].size(0)
      for batch_idx in range(num_batchs):
        # in val: 2007_002094 bird, 2007_000129 bike
        # 2007_003205 baby, 2007_000733 kid, 2007_007470 plane
        if (num_batchs == 1) and (image_name[batch_idx] == '2007_000129'):
          plt.figure()
          plt.subplot(231)
          plt.imshow(lower_level_feat_save[0][batch_idx].sum(2).detach().cpu().numpy())
          plt.subplot(232)
          plt.imshow(lower_level_feat_save[1][batch_idx].sum(2).detach().cpu().numpy())
          plt.subplot(233)
          plt.imshow(lower_level_feat_save[2][batch_idx].sum(2).detach().cpu().numpy())
          plt.subplot(234)
          plt.imshow(lower_level_feat_save[3][batch_idx].sum(2).detach().cpu().numpy())
          plt.subplot(235)
          plt.imshow(lower_level_feat_save[4][batch_idx].sum(2).detach().cpu().numpy())
          plt.show()

          scio.savemat('data/feats/{}.mat'.format(image_name[batch_idx]),
                       {'feat_0': (lower_level_feat_save[0][batch_idx] * 100).byte().detach().cpu().numpy(),
                        'feat_1': (lower_level_feat_save[1][batch_idx] * 100).byte().detach().cpu().numpy(),
                        'feat_2': (lower_level_feat_save[2][batch_idx] * 100).byte().detach().cpu().numpy(),
                        'feat_3': (lower_level_feat_save[3][batch_idx] * 100).byte().detach().cpu().numpy(),
                        'feat_4': (lower_level_feat_save[4][batch_idx] * 10).byte().detach().cpu().numpy()})

    label_context = None
    scores_small = scores
    scores_small_mp = scores
    edge_weight_dirs = edge_weights
    edge_map = None

    if self.args.enable_pairwise_net:
      assert not self.args.deeplab_enable_interpolation

      # cost_volume of edge:(batch,(n_dir+1)/2,h,w) from left->right, top->down
      batch, _, h, w = feat_before_last_conv.size()
      edge_volume = inputs.new_zeros((batch, self.edge_volume_number, h, w, self.args.n_classes, self.args.n_classes))

      for idx in range(self.edge_volume_number):
        # feat_before_last_conv:(batch,304,h/4,w/4)
        feat_before_last_conv_shift = feat_before_last_conv.new_zeros(feat_before_last_conv.size())
        if idx == 0: # left->right
          feat_before_last_conv_shift[:, :, 1:, :] = feat_before_last_conv[:, :, :-1, :]
        elif idx == 1: # top->down
          feat_before_last_conv_shift[:, :, :, 1:] = feat_before_last_conv[:, :, :, :-1]
        elif idx == 2: # left+up->right+down
          feat_before_last_conv_shift[:, :, 1:, 1:] = feat_before_last_conv[:, :, :-1, :-1]
        elif idx == 3: # right+up->left+down
          feat_before_last_conv_shift[:, :, 1:, :-1] = feat_before_last_conv[:, :, :-1, 1:]
        else:
          assert False, 'Error! Only 8 directions are supported.'

        edge_volume_per = torch.cat([feat_before_last_conv, feat_before_last_conv_shift], dim=1)
        edge_volume[:, idx] = self.pairwise_net(edge_volume_per).permute(0,2,3,1) \
          .view(batch,h,w,self.args.n_classes,self.args.n_classes).contiguous()

      costs = -scores
      results = self.mp_layer(costs.unsqueeze(1), pairwise_terms=edge_volume)
      scores = -results[0].squeeze(1)
      scores_small_mp = scores

      scores = F.interpolate(scores, size=input_size, mode='bilinear', align_corners=True)
      score_org = F.interpolate(score_org, size=input_size, mode='bilinear', align_corners=True)
    else:
      if not self.args.deeplab_enable_interpolation:
        if self.enable_mp_layer:
          if self.args.edge_mode in ['edge_net', 'edge_net_sigmoid']:
            # for feat_idx in range(1, len(lower_level_feats)):
            #   lower_level_feats[feat_idx] = F.interpolate(lower_level_feats[feat_idx],
            #                                               size=lower_level_feats[0].size()[-2:],
            #                                               mode='bilinear',
            #                                               align_corners=True)
            # lower_level_feats = torch.cat(lower_level_feats, dim=1)
            edge_map = self.edge_net(lower_level_feats)

            # Shifting edge map by directions
            # edge_weight_dirs:(batch,dir,h,w)
            edge_weight_dirs = getEdgeShift(self.args.edge_mode,
                                            edge_map.squeeze(1),
                                            self.args.mpnet_n_dirs)
            edge_weight_dirs = edge_weight_dirs.permute(1, 0, 2, 3).contiguous()

          if (edge_weight_dirs is not None) and (scores.size()[-2:] != edge_weight_dirs.size()[-2:]):
            scores = F.interpolate(scores, size=edge_weight_dirs.size()[2:], mode='bilinear', align_corners=True)

          scores_small = scores
          costs = -scores
          costs = costs - costs.min(1, keepdim=True)[0]

          if self.enable_score_scale:
            costs = costs / costs.max(1, keepdim=True)[0]

          results = self.mp_layer(costs.unsqueeze(1), edge_weights=edge_weight_dirs)
          scores = -results[0].squeeze(1)
          scores_small_mp = scores
          label_context = results[1]

        if scores.size()[-2:] != input_size:
          scores = F.interpolate(scores, size=input_size, mode='bilinear', align_corners=True)

        if score_org.size()[-2:] != input_size:
          score_org = F.interpolate(score_org, size=input_size, mode='bilinear', align_corners=True)

        if edge_map is not None:
          if edge_map.size()[-2:] != input_size:
            edge_map = F.interpolate(edge_map, size=input_size, mode='nearest').squeeze(1)

    if self.enable_score_scale:
      scores = nn.LayerNorm(scores.size()[1:], eps=0, elementwise_affine=False)(scores)

    return [score_org, scores, edge_map, label_context, scores_small, scores_small_mp, edge_weight_dirs]

  def get_1x_lr_params(self):
    modules = [self.deeplab.backbone]

    for i in range(len(modules)):
      for m in modules[i].modules():
        if isinstance(m, nn.Conv2d) \
                or isinstance(m, nn.ConvTranspose2d):
          for p in m.parameters():
            if p.requires_grad:
              yield p

        # Unnecessary to freeze on single GPU
        if not self.args.deeplab_freeze_bn:
          if isinstance(m, SynchronizedBatchNorm2d) \
                  or isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
              if p.requires_grad:
                yield p

  def get_10x_lr_params(self):
    if self.args.enable_fix_unary:
      modules = []
    else:
      if self.args.disable_aspp:
        modules = [self.deeplab.decoder]
      else:
        modules = [self.deeplab.aspp, self.deeplab.decoder]

    if self.args.enable_pairwise_net:
      modules += [self.pairwise_net]

    if self.args.edge_mode in ['edge_net', 'edge_net_sigmoid']:
      modules += [self.edge_net]

    for i in range(len(modules)):
      for m in modules[i].modules():
        if isinstance(m, nn.Conv2d) \
                or isinstance(m, nn.ConvTranspose2d):
          for p in m.parameters():
            if p.requires_grad:
              yield p

        # Unnecessary to freeze on single GPU
        if not self.args.deeplab_freeze_bn:
          if isinstance(m, SynchronizedBatchNorm2d) \
                  or isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
              if p.requires_grad:
                yield p

  def get_05x_lr_params(self):
    if self.args.mpnet_smoothness_train in ['softmax', 'sigmoid', 'ON']:
      for p in self.mp_layer.parameters():
        if p.requires_grad:
          yield p

  def freezebn_modules(self, modules):
    if not isinstance(modules, list):
      modules = [modules]

    for i in range(len(modules)):
      for m in modules[i].modules():
        if isinstance(m, SynchronizedBatchNorm2d) \
                or isinstance(m, nn.BatchNorm2d):
          m.eval()  # !!! eval mode will fix running_mean and running_var
