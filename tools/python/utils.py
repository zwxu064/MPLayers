import os
import json
import torch
import numpy as np
import math
import numbers
import random
import torchvision.transforms as transforms

from torch import nn
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt


def remove_tree(fold_dir, wildcards=None):
  for root, dirs, files in os.walk(fold_dir):
    for file in files:
      if wildcards is not None:
        if (file.split('.')[-1] not in wildcards.split('.')[-1]):
          continue
      os.remove(os.path.join(root, file))


##### display and output log related #####
def parse_json(file_path):
  if not os.path.exists(file_path):
    raise FileNotFoundError

  with open(file_path, 'r') as config:
    data = json.load(config)

  return data


def check_dir(dir_name):
  if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def print_info(obj):
  obj_len = len(obj)
  string = ''

  for i, key in zip(range(obj_len), obj.keys()):
    if i < obj_len - 1:
      if obj[key] == 'best':
        string += '[={}] '.format(obj[key])
      elif obj[key] in {'train', 'valid'}:
        string += '[{}] '.format(obj[key])
      else:
        if isinstance(obj[key], (float, np.float64, np.float32)):
          string += '{}: {:.4f}, '.format(key, obj[key]).rstrip('0').rstrip('.')
        else:
          string += '{}: {}, '.format(key, obj[key])
    elif i == obj_len - 1:
      if isinstance(obj[key], (float, np.float64, np.float32)):
        string += '{}: {:.4f}'.format(key, obj[key]).rstrip('0').rstrip('.')
      else:
        string += '{}: {}'.format(key, obj[key])

  print(string)


def save_disp_image(img_path, img, scale, mode='grey'):
  assert (img_path is not None)

  if mode in {'grey', 'gray'}:
    assert (scale > 0)
    if len(img.size()) == 3:
      img = img.squeeze(0)

    img_scale = np.round(
      np.multiply(img.cpu().data.numpy(), scale.data.numpy()))
    img_pil = Image.fromarray(img_scale.astype(np.uint8))
    img_pil.save(img_path, "PNG")
  elif mode == 'rgb':
    if len(img.size()) == 4:
      img = img.squeeze(0)

    if img.size()[0] == 3:
      img = np.transpose(img.cpu().data.numpy().astype(np.uint8), (1, 2, 0))

    img_pil = Image.fromarray(img, 'RGB')
    img_pil.save(img_path, "JPEG")


def write_to_tensorboard(tb_mode, writer, obj):
  assert (tb_mode in {'scalar', 'image'})

  for key in obj.keys():
    if tb_mode == 'scalar':
      if key in {'train_loss', 'valid_loss', 'acc', 'acc_acl',
                 'mean_iou', 'fwavacc', 'loss', 'error', 'train_error',
                 'valid_error', 'train_acc0.5', 'train_acc1', 'train_acc2',
                 'train_acc3', 'valid_acc0.5', 'valid_acc1', 'valid_acc2',
                 'valid_acc3', 'train_acc5', 'valid_acc5'}:
        net_mode = obj['mode']

        if 'batch_iter' in obj.keys():
          writer.add_scalar(key, obj[key], obj['batch_iter'])
        else:
          writer.add_scalar(key, obj[key], obj['epoch'])
    elif tb_mode == 'image':
      writer.add_image(key, obj[key])


##### calculation related #####
# (pred, gt)->(h, w) from {0, ..., 20}
def fast_hist(label_pred, label_true, num_class):
  mask = (label_true >= 0) & (label_true < num_class)
  hist = np.bincount(
    num_class * label_true[mask].astype(int) +
    label_pred[mask], minlength=num_class ** 2).reshape(
    num_class, num_class)
  return hist


def get_upsampling_weight(in_channels, out_channels, kernel_size):
  factor = (kernel_size + 1) // 2
  if kernel_size % 2 == 1:
    center = factor - 1
  else:
    center = factor - 0.5
  og = np.ogrid[:kernel_size, :kernel_size]
  filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
  weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                    dtype=np.float)
  weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt

  return torch.from_numpy(weight)


def get_adaptive_crop_size(data_large, data_small, crop_size):
  large_size = data_large.size()
  small_size = data_small.size()
  h_diff = large_size[-2] - small_size[-2]
  w_diff = large_size[-1] - small_size[-1]

  assert (len(large_size) == len(large_size))
  assert (h_diff >= 0 and w_diff >= 0)

  if not isinstance(crop_size, tuple):
    crop_size = (crop_size, crop_size)

  crop_top = min(crop_size[0], h_diff)
  crop_left = min(crop_size[1], w_diff)

  return ((crop_top, h_diff - crop_top), (crop_left, w_diff - crop_left))


def evaluate(predictions, gts, class_num):
  hist = np.zeros((class_num, class_num))

  for lp, lt in zip(predictions, gts):
    hist += fast_hist(lp.flatten(), lt.flatten(), class_num)

  with np.errstate(divide='ignore', invalid='ignore'):  # divide 0 ignored
    # axis 0: gt, axis 1: prediction
    acc = np.divide(np.diag(hist).sum(), hist.sum())
    acc_cls = np.divide(np.diag(hist), hist.sum(axis=1))
    acc_cls = np.nanmean(acc_cls)

    iu = np.divide(np.diag(hist), (hist.sum(axis=1)) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iu)

    freq = np.divide(hist.sum(axis=1), hist.sum())
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

  return acc, acc_cls, mean_iou, fwavacc


def cal_accuracy(predict, gt, threshold):
  img_size = predict.size()
  diff = np.absolute(np.subtract(predict.cpu().data.numpy(), gt.cpu().data.numpy()))
  if len(img_size) == 3:
    pixel_num = img_size[1] * img_size[2]
  elif len(img_size) == 1:
    pixel_num = img_size[0]
  else:
    pixel_num = 1

  acc = np.sum(np.less_equal(diff, threshold)) / pixel_num
  return 100 * acc


def weight_init(model):
  for ind, module in enumerate(model.modules()):
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
      if module.weight is not None:
        nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2))

      if module.bias is not None:
        nn.init.constant_(module.bias, 0)

    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
      if module.weight is not None:
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    if isinstance(module, nn.ConvTranspose2d):
      nn.init.constant_(module.weight, 0)
      # assert module.kernel_size[0] == module.kernel_size[1]
      # module.weight.data.copy_(get_upsampling_weight(module.in_channels,
      #                                                module.out_channels,
      #                                                module.kernel_size[0]))


def weight_fixed(obj):
  for i in range(len(obj)):
    for param in obj[i].parameters():
      param.requires_grad = False


def check_valid(input, enable_zero=False):
  if torch.isnan(input).any() or torch.isinf(input).any():
    status = False
  elif enable_zero and (input == 0).any():
    status = False
  else:
    status = True

  return status


def to_tensor_custom(input, dtype=torch.float32, device='cpu'):
  return torch.from_numpy(input).type(dtype).to(device)


def colorize_mask(mask, palette):
  new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
  new_mask.putpalette(palette)

  return new_mask


def visualize_context(context):
  context = context.float().cpu().detach()
  context = context * 255 // context.max()
  context = Image.fromarray(context.numpy())

  return context

def get_params_per(idx, out_channels, kernel_sizes, strides, paddings):
  if isinstance(out_channels, list):
    out_channel = out_channels[idx]
  else:
    out_channel = out_channels

  if isinstance(kernel_sizes, list):
    assert len(kernel_sizes) == len(out_channels)
    kernel_size = kernel_sizes[idx]
  else:
    kernel_size = kernel_sizes

  if isinstance(strides, list):
    assert len(strides) == len(out_channels)
    stride = strides[idx]
  else:
    stride = strides

  if isinstance(paddings, list):
    assert len(paddings) == len(out_channels)
    padding = paddings[idx]
  else:
    padding = paddings

  return out_channel, kernel_size, stride, padding


# Check if use DataParallel for model
def get_model_state_dict(model):
  if hasattr(model, 'module'):
    model_state_dict = model.module.state_dict()
  else:
    model_state_dict = model.state_dict()

  return model_state_dict


def load_optimizer_state_dict(checkpoint, optimizer, enable_cuda=True):
  optimizer.load_state_dict(checkpoint)
  for state in optimizer.state.values():
    for k, v in state.items():
      if isinstance(v, torch.Tensor):
        if enable_cuda:
          state[k] = v.cuda()
        else:
          state[k] = v.cpu()

  return optimizer


def check_data_input(inputs, time=5):
  is_list = isinstance(inputs, list)
  inputs = [inputs] if not is_list else inputs
  count, n_inputs = 1, len(inputs)
  rows = math.floor(math.sqrt(n_inputs))
  cols = math.ceil(n_inputs / rows)
  fig = plt.figure(figsize=(rows, cols))

  for input in inputs:
    if input is not None:
      fig.add_subplot(rows, cols, count)
      count += 1
      input_size = input.size()

      if len(input_size) == 4:
        if input.size()[1] == 1:
          plt.imshow(input[0, 0, :, :])
        else:
          plt.imshow(input[0, :, :, :].permute(1, 2, 0))
      elif len(input_size) == 3:
        plt.imshow(input[0, :, :])

  if time is not None:
    plt.show(block=False)
    plt.pause(time)
    plt.close()
  else:
    plt.show()


def adjust_learning_rate(optimizer, epoch, step):
  if (epoch > 1) and (epoch % step == 0):
    for param_group in optimizer.param_groups:
      param_group['lr'] *= 0.1
    print('Adjust learning rate to {:e}'.format(optimizer.param_groups[0]['lr']))


def split_weight_bias(model, mode='weight'):
  return [p[1] for p in list(filter(lambda p: (mode in p[0]) and (p[1].requires_grad),
                                    model.named_parameters()))]


##### update evaluate parameters #####
class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class FlipChannels(object):
  def __call__(self, img):
    img = np.array(img)[:, :, ::-1]
    return Image.fromarray(img.astype(np.uint8))


class MaskToTensor(object):
  def __init__(self, enable_long=True):
    self.enable_long = enable_long

  def __call__(self, mask):
    if mask is not None:
      mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32))
      mask_tensor = mask_tensor.long() if self.enable_long else mask_tensor
      return mask_tensor
    else:
      return None

class GTToTensor(object):
  def __init__(self, enable_long=False):
    self.enable_long = enable_long

  def __call__(self, mask, scale=1, use_round=False):
    if mask is not None:
      mask_scale = np.divide(np.array(mask, dtype=np.float32), scale)
      mask_scale = np.round(mask_scale) if use_round else mask_scale
      mask_scale = torch.from_numpy(mask_scale)
      mask_scale = mask_scale.long() if self.enable_long else mask_scale
      return mask_scale
    else:
      return None


class DeNormalize(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, tensor):
    for t, m, s in zip(tensor, self.mean, self.std):
      t.mul_(s).add_(m)
    return tensor


class SlidingCrop(object):
  def __init__(self, crop_size, stride_rate, ignore_label):
    self.crop_size = crop_size
    self.stride_rate = stride_rate
    self.ignore_label = ignore_label

  def _pad(self, img, mask):
    h, w = img.shape[: 2]
    pad_h = max(self.crop_size - h, 0)
    pad_w = max(self.crop_size - w, 0)
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant',
                  constant_values=self.ignore_label)
    return img, mask, h, w

  def __call__(self, img, mask):
    assert img.size == mask.size

    w, h = img.size
    long_size = max(h, w)
    img = np.array(img)
    mask = np.array(mask)

    if long_size > self.crop_size:
      stride = int(math.ceil(self.crop_size * self.stride_rate))
      h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
      w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
      img_slices, mask_slices, slices_info = [], [], []
      for yy in range(h_step_num):
        for xx in range(w_step_num):
          sy, sx = yy * stride, xx * stride
          ey, ex = sy + self.crop_size, sx + self.crop_size
          img_sub = img[sy: ey, sx: ex, :]
          mask_sub = mask[sy: ey, sx: ex]
          img_sub, mask_sub, sub_h, sub_w = self._pad(img_sub, mask_sub)
          img_slices.append(Image.fromarray(img_sub.astype(np.uint8))
                                           .convert('RGB'))
          mask_slices.append(Image.fromarray(mask_sub.astype(np.uint8))
                                            .convert('P'))
          slices_info.append([sy, ey, sx, ex, sub_h, sub_w])
      return img_slices, mask_slices, slices_info
    else:
      img, mask, sub_h, sub_w = self._pad(img, mask)
      img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
      mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
      return [img], [mask], [[0, sub_h, 0, sub_w, sub_h, sub_w]]


class RandomCrop(object):
  def __init__(self, size=None):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, inputs):
    is_list = isinstance(inputs, list)
    inputs = [inputs] if not is_list else inputs
    outputs = []
    w, h = inputs[0].size

    if self.size is None:
      th, tw = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
    else:
      th, tw = self.size

    w_gap, h_gap = tw - w, th - h
    w_rand, h_rand = random.randint(0, max(w - tw, 0)), random.randint(0, max(h - th, 0))
    w_0 = (w_gap // 2) if (w - tw < 0) else -w_rand
    h_0 = (h_gap // 2) if (h - th < 0) else -h_rand
    border = (w_0, h_0, w_gap - w_0, h_gap - h_0)

    for input in inputs:
      if input is not None:
        # border:(left,up,right,down)
        fill_v = 0 if input.mode == 'RGB' else 255
        outputs.append(ImageOps.expand(input, border=border, fill=fill_v))
      else:
        outputs.append(None)

    if not is_list:
      return outputs[0]
    else:
      return outputs


class RandomHorizontalFlip(object):
  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, inputs):
    is_list = isinstance(inputs, list)
    outputs = []
    inputs = [inputs] if not is_list else inputs
    enable_flip = np.random.random() < self.p

    if enable_flip:
      for input in inputs:
        if input is not None:
          outputs.append(transforms.RandomHorizontalFlip(p=1.)(input))
        else:
          outputs.append(None)
    else:
      outputs = inputs

    if not is_list:
      return outputs[0]
    else:
      return outputs


class RandomVerticalFlip(object):
  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, inputs):
    is_list = isinstance(inputs, list)
    outputs = []
    inputs = [inputs] if not is_list else inputs
    enable_flip = np.random.random() < self.p

    if enable_flip:
      for input in inputs:
        if input is not None:
          outputs.append(transforms.RandomVerticalFlip(p=1.)(input))
        else:
          outputs.append(None)
    else:
      outputs = inputs

    if not is_list:
      return outputs[0]
    else:
      return outputs


class CenterCrop(object):
  def __init__(self, size=None, mode='left_up'):
    self.mode = mode
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, inputs):
    is_list = isinstance(inputs, list)
    outputs = []
    inputs = [inputs] if not is_list else inputs
    w, h = inputs[0].size

    if self.size is None:
      th, tw = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
    else:
      th, tw = self.size

    w_gap, h_gap = tw - w, th - h
    w_0, h_0 = w_gap // 2, h_gap // 2
    border = (w_0, h_0, w_gap - w_0, h_gap - h_0)

    for input in inputs:
      if input is not None:
        # border:(left,up,right,down)
        fill_v = 0 if input.mode == 'RGB' else 255
        outputs.append(ImageOps.expand(input, border=border, fill=fill_v))
      else:
        outputs.append(None)

    if not is_list:
      return outputs[0]
    else:
      return outputs


class RandomGaussianBlur(object):
  def __call__(self, inputs):
    is_list = isinstance(inputs, list)
    outputs = []
    inputs = [inputs] if not is_list else inputs

    for input in inputs:
      if (input.mode == 'RGB') and (random.random() < 0.5):
         outputs.append(input.filter(ImageFilter.GaussianBlur(radius=random.random())))
      else:
         outputs.append(input)

    if not is_list:
      return outputs[0]
    else:
      return outputs


class CrossEntropyLoss2D(nn.Module):
  def __init__(self, weight=None, reduction='mean', ignore_index=255, reg=0, eps=1e-10):
    super(CrossEntropyLoss2D, self).__init__()
    self.nll_loss = nn.NLLLoss(weight, reduction=reduction, ignore_index=ignore_index)
    self.reg = reg
    self.eps = eps

  # inputs:(batch,n_labels,n_labels,2,height,width) to (batch,n_labels*n_labels,2,height,width)
  def forward(self, predicts, targets, model=None):
    if len(predicts.size()) == 6:
      batch, _, _, n_dir_type, height, width = predicts.size()
      predicts = predicts.reshape(batch, -1, n_dir_type, height, width)

    loss = self.nll_loss((predicts + self.eps).log(), targets)

    param_loss = 0
    if (self.reg > 0) and (model is not None):
      for m in model.parameters():
        # param_loss += m.norm(2)
        param_loss += m.pow(2).sum()

    return loss + self.reg * param_loss


class StereoLoss(nn.Module):
  def __init__(self, loss_type='smooth_l1', reduction='mean', reg=1e-4):
    super(StereoLoss, self).__init__()
    if loss_type == 'smooth_l1':
      self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
    elif loss_type == 'l1':
      self.loss_fn = nn.L1Loss(reduction=reduction)
    else:
      self.loss_fn = nn.MSELoss(reduction=reduction)

    self.reg = reg

  def forward(self, predicts, targets, model=None):
    loss = self.loss_fn(predicts, targets)
    param_loss = 0

    if (self.reg > 0) and (model is not None):
      for m in model.parameters():
        # param_loss += m.norm(2)
        param_loss += m.pow(2).sum()

    return loss + self.reg * param_loss
