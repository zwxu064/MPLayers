###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import sys
import os
import torch
import torchvision.utils as vutils

sys.path.append('..')
from dataloaders.utils import decode_segmap
from PIL import Image
from skimage import io


class DeNormalize(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, tensor):
    for t, m, s in zip(tensor, self.mean, self.std):
      t.mul_(s).add_(m)

    return tensor


def apply_mask(image, mask, color):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] + color[c],
                                  image[:, :, c])
    return image

def visualize_prediction(dataset, path, pred):
    n, h, w = pred.shape
    image = np.zeros((h, w, 3))
    dataset = dataset.lower()
    # image = image.astype(np.uint32)

    if dataset == 'cityscapes':
      colors = [[128, 64, 128],
               [244, 35, 232],
               [70, 70, 70],
               [102, 102, 156],
               [190, 153, 153],
               [153, 153, 153],
               [250, 170, 30],
               [220, 220, 0],
               [107, 142, 35],
               [152, 251, 152],
               [70, 130, 180],
               [220, 20, 60],
               [255, 0, 0],
               [0, 0, 142],
               [0, 0, 70],
               [0, 60, 100],
               [0, 80, 100],
               [0, 0, 230],
               [119, 11, 32]]
    else:
      assert dataset == 'sbd'
      colors = [[128, 0, 0],
               [0, 128, 0],
               [128, 128, 0],
               [0, 0, 128],
               [128, 0, 128],
               [0, 128, 128],
               [128, 128, 128],
               [64, 0, 0],
               [192, 0, 0],
               [64, 128, 0],
               [192, 128, 0],
               [64, 0, 128],
               [192, 0, 128],
               [64, 128, 128],
               [192, 128, 128],
               [0, 64, 0],
               [128, 64, 0],
               [0, 192, 0],
               [128, 192, 0],
               [0, 64, 128]]

    pred = np.where(pred >= 0.5, 1, 0).astype(np.bool)
    edge_sum = np.zeros((h, w))

    for i in range(n):
      color = colors[i]
      edge = pred[i,:,:]
      edge_sum = edge_sum + edge
      masked_image = apply_mask(image, edge, color)

    edge_sum = np.array([edge_sum, edge_sum, edge_sum])
    edge_sum = np.transpose(edge_sum, (1, 2, 0))
    idx = edge_sum > 0
    masked_image[idx] = masked_image[idx]/edge_sum[idx]
    masked_image[~idx] = 255
    
    io.imsave(path, masked_image/255)


def visualization(image, target, pred, edge=None, image_name=None,
                  accuracy=None, save_dir=None, enable_save_all=False):
  mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  image_per = (DeNormalize(*mean_std)(image).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
  target_per = (decode_segmap(target, 'pascal') * 255).astype(np.uint8)
  segmap = (decode_segmap(pred, 'pascal') * 255).astype(np.uint8)

  if edge is None:
    nrow = 3
    img_list = [transforms.ToTensor()(Image.fromarray(image_per).convert('RGB')),
                transforms.ToTensor()(Image.fromarray(target_per).convert('RGB')),
                transforms.ToTensor()(Image.fromarray(segmap).convert('RGB'))]
  else:
    nrow = 2
    edge = edge.unsqueeze(2).repeat(1, 1, 3).cpu().numpy().astype(np.float32)
    edge = edge / edge.max() * 255
    edge = edge.astype(np.uint8)

    # Patch
    edge_h, edge_w, _ = edge.shape
    img_h, img_w, _ = image_per.shape
    edge_org = edge

    if (edge_h != img_h) or (edge_w != edge_w):
      edge = np.zeros((img_h, img_w, 3), dtype=np.uint8)
      edge[:edge_h, :edge_w, :] = edge_org

    img_list = [transforms.ToTensor()(Image.fromarray(image_per).convert('RGB')),
                transforms.ToTensor()(Image.fromarray(target_per).convert('RGB')),
                transforms.ToTensor()(Image.fromarray(segmap).convert('RGB')),
                transforms.ToTensor()(Image.fromarray(edge).convert('RGB'))]

  valid_visual = torch.stack(img_list, 0)
  valid_visual = vutils.make_grid(valid_visual, nrow=nrow, padding=0)

  if (image_name is not None) and (accuracy is not None) and (save_dir is not None):
    save_path = os.path.join(save_dir, '{}_{:.2f}.png'.format(image_name, accuracy))
    vutils.save_image(valid_visual, save_path)

    if enable_save_all:
      save_dir = os.path.join(save_dir, image_name)
      os.makedirs(save_dir) if (not os.path.exists(save_dir)) else None
      vutils.save_image(img_list[0], os.path.join(save_dir, 'RGB.png'))
      vutils.save_image(img_list[1], os.path.join(save_dir, 'GT.png'))
      vutils.save_image(img_list[2], os.path.join(save_dir, 'Final_Seg.png'))

      if edge is not None:
        edge_save = transforms.ToTensor()(Image.fromarray(edge_org).convert('RGB'))
        vutils.save_image(edge_save, os.path.join(save_dir, 'Edge.png'))


def visualization_dff(image, side5, fuse, save_path=None):
  mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  image_per = (DeNormalize(*mean_std)(image).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
  side5_per = side5.cpu().numpy().astype(np.uint8)
  fuse_per = fuse.cpu().numpy().astype(np.uint8)

  img_list = [transforms.ToTensor()(Image.fromarray(image_per).convert('RGB')),
              transforms.ToTensor()(Image.fromarray(side5_per).convert('RGB')),
              transforms.ToTensor()(Image.fromarray(fuse_per).convert('RGB'))]
  valid_visual = torch.stack(img_list, 0)
  valid_visual = vutils.make_grid(valid_visual, nrow=3, padding=0)

  if save_path is not None:
    vutils.save_image(valid_visual, save_path)


def visualization_png(edge_hor, edge_ver, pred, pred_mp, image_name=None,
                      accuracy=None, save_dir=None, enable_save_all=False):
  edge_hor = edge_hor.unsqueeze(2).repeat(1, 1, 3).cpu().numpy().astype(np.float32)
  edge_ver = edge_ver.unsqueeze(2).repeat(1, 1, 3).cpu().numpy().astype(np.float32)
  segmap = (decode_segmap(pred, 'pascal') * 255).astype(np.uint8)
  segmap_mp = (decode_segmap(pred_mp, 'pascal') * 255).astype(np.uint8)
  edge_hor = edge_hor / edge_hor.max() * 255
  edge_ver = edge_ver / edge_ver.max() * 255
  edge_hor, edge_ver = edge_hor.astype(np.uint8), edge_ver.astype(np.uint8)

  img_list = [transforms.ToTensor()(Image.fromarray(edge_hor).convert('RGB')),
              transforms.ToTensor()(Image.fromarray(edge_ver).convert('RGB')),
              transforms.ToTensor()(Image.fromarray(segmap).convert('RGB')),
              transforms.ToTensor()(Image.fromarray(segmap_mp).convert('RGB'))]
  valid_visual = torch.stack(img_list, 0)
  valid_visual = vutils.make_grid(valid_visual, nrow=2, padding=0)

  if (image_name is not None) and (accuracy is not None) and (save_dir is not None):
    save_path = os.path.join(save_dir, '{}_edge_{:.2f}.png'.format(image_name, accuracy))
    vutils.save_image(valid_visual, save_path)

    if enable_save_all:
      save_dir = os.path.join(save_dir, image_name)
      os.makedirs(save_dir) if (not os.path.exists(save_dir)) else None
      vutils.save_image(img_list[0], os.path.join(save_dir, 'Edge_Hor.png'))
      vutils.save_image(img_list[1], os.path.join(save_dir, 'Edge_Ver.png'))
      vutils.save_image(img_list[2], os.path.join(save_dir, 'Unary_Seg_Scaled.png'))
      vutils.save_image(img_list[3], os.path.join(save_dir, 'Final_Seg_Scaled.png'))