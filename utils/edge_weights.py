import torch
import torch.nn.functional as F


def get_steps(dir):
  if dir == 0:
    h_step, w_step = 0, 1
  elif dir == 1:
    h_step, w_step = 0, -1
  elif dir == 2:
    h_step, w_step = 1, 0
  elif dir ==3:
    h_step, w_step = -1, 0
  elif dir == 4:
    h_step, w_step = 1, 1
  elif dir == 5:
    h_step, w_step = -1, -1
  elif dir == 6:
    h_step, w_step = 1, -1
  elif dir == 7:
    h_step, w_step = -1, 1
  elif dir == 8:
    h_step, w_step = 1, 2
  elif dir == 9:
    h_step, w_step = -1, -2
  elif dir == 10:
    h_step, w_step = 2, 1
  elif dir == 11:
    h_step, w_step = -2, -1
  elif dir == 12:
    h_step, w_step = 2, -1
  elif dir == 13:
    h_step, w_step = -2, 1
  elif dir == 14:
    h_step, w_step = 1, -2
  elif dir == 15:
    h_step, w_step = -1, 2
  else:
    print('Warning, edge weights support only 16 directions by far')
    h_step, w_step = 0, 0

  return h_step, w_step


def getEdgeShift(edge_mode, image, n_dirs, sigma=10, threshold=0, penalty=0):
  # image size: (batch,channels,h,w), edge_weights:(n_dir,batch,channels,h,w)
  image_size_len = len(image.size())

  if image_size_len == 2:
    image = image.unsqueeze(0).unsqueeze(0)
  elif image_size_len == 3:
    image = image.unsqueeze(0)

  assert len(image.size()) == 4, image.size()
  batch, channels, height, width = image.size()
  img_scale_shift = image.new_zeros(n_dirs, batch, channels, height, width)

  for dir in range(n_dirs):
    h_step, w_step = get_steps(dir)

    if h_step > 0:
      src_h_start, src_h_stop = 0, height - h_step
      tar_h_start, tar_h_stop = h_step, height
    else:
      src_h_start, src_h_stop = -h_step, height
      tar_h_start, tar_h_stop = 0, height + h_step

    if w_step > 0:
      src_w_start, src_w_stop = 0, width - w_step
      tar_w_start, tar_w_stop = w_step, width
    else:
      src_w_start, src_w_stop = -w_step, width
      tar_w_start, tar_w_stop = 0, width + w_step

    img_patch_shifted = image[:, :, src_h_start:src_h_stop, src_w_start:src_w_stop]
    img_patch = image[:, :, tar_h_start:tar_h_stop, tar_w_start:tar_w_stop]
    img_patch_diff = (img_patch - img_patch_shifted).abs()

    # Necessary for 255 edges with more than two pixels
    if edge_mode == 'gt_edge':
      img_patch_diff[img_patch == 255] = 1
    elif edge_mode in {'canny'}:
      img_patch_diff[img_patch == 1] = 1

    if edge_mode == 'gt_edge':
      grad = (img_patch_diff == 0).float()
    elif edge_mode == 'superpixel_edge':
      grad = (img_patch_diff.sum(1) == 0).float()
    elif edge_mode in {'canny', 'sobel'}:
      grad = 1 - img_patch_diff
    elif edge_mode == 'edge_net_sigmoid':
      grad = 1 - img_patch_diff  # 2 * nn.Sigmoid()(-img_patch_diff) close to edge net
    elif edge_mode == 'edge_net':
      grad = (-img_patch_diff).exp()
    elif edge_mode in ['kernel_cue_real', 'kernel_cue_binary']:
      grad = torch.exp(torch.mean(torch.pow(img_patch_diff, 2), dim=1) / (-2 * sigma ** 2))
      if edge_mode == 'kernel_cue_binary':
        grad = (grad >= 0.8 * grad.mean()).float()
    elif edge_mode == 'threshold':
      grad = (torch.mean(torch.pow(img_patch_diff, 2), dim=1) < threshold).float() * penalty
    else:
      grad = img_patch_shifted

    img_scale_shift[dir, :, :, tar_h_start:tar_h_stop, tar_w_start:tar_w_stop] = grad

  if image_size_len == 2:
    img_scale_shift = img_scale_shift.squeeze(1).squeeze(1)
  elif image_size_len == 3:
    img_scale_shift = img_scale_shift.squeeze(1)

  return img_scale_shift


# Get edge weights from ground truth
def multi_edge_weights(edge_mode,
                       img,
                       n_dirs,
                       scale_list=None,
                       sigma=5,
                       threshold=10,
                       penalty=2,
                       padding_mask=None):
  img_group_scale, padding_mask_scale = [], []
  channel, height, width = img.size()
  enable_padding_mask = padding_mask is not None

  # ==== Downsample
  for scale in scale_list:
    h_size, w_size = int(height * scale), int(width * scale)

    if scale == 1:
      img_group_scale.append(img)
      if enable_padding_mask:
        padding_mask_scale.append(padding_mask)
    else:
      w_grid = torch.linspace(-1, 1, w_size).repeat(h_size, 1)
      h_grid = torch.linspace(-1, 1, h_size).view(-1, 1).repeat(1, w_size)
      grid = torch.cat((w_grid.unsqueeze(2), h_grid.unsqueeze(2)), 2)

      if False:
        img_scale = F.grid_sample(img.unsqueeze(0), grid.unsqueeze(0))
      else:
        inter_mode = 'nearest' if edge_mode in ['gt_edge', 'edge_net', 'edge_net_sigmoid', 'superpixel_edge'] else 'bilinear'
        enable_align_corners = None if (inter_mode == 'nearest') else True
        img_scale = F.interpolate(img.unsqueeze(0), scale_factor=(scale, scale),
                                  mode=inter_mode, align_corners=enable_align_corners)

      img_group_scale.append(img_scale.squeeze(0))

      if enable_padding_mask:
        padding_mask_tmp = F.grid_sample(padding_mask.unsqueeze(0), grid.unsqueeze(0))
        padding_mask_scale.append((padding_mask_tmp.squeeze(0) > 0.5).float())

  # ==== Dirs
  # img_scale:(channels,h,w)
  # img_scale_shift:(n_dir,h,w)
  img_group_scale_shift = []

  for idx, img_scale in enumerate(img_group_scale):
    if n_dirs > 0:
      # Shifting edge map by directions
      img_scale_shift = getEdgeShift(edge_mode,
                                     img_scale,
                                     n_dirs,
                                     sigma=sigma,
                                     threshold=threshold,
                                     penalty=penalty)
      img_scale_shift = img_scale_shift.squeeze(1)
    else:
      img_scale_shift = img_scale

    if enable_padding_mask:
      img_scale_shift = img_scale_shift * padding_mask_scale[idx]

    if edge_mode == 'superpixel_edge':
      img_scale_shift = img_scale_shift[:, 0]

    img_group_scale_shift.append(img_scale_shift)

  return img_group_scale_shift