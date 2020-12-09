import torch

# Must set this as value type, otherwise the torch_version_major == 0 will be wrong
torch_version_major = int(torch.__version__.split('.')[0])

def message_norm(message, loc, dir):
  assert len(loc) == 2
  h, w = loc
  n_labels = message.size()[-1]
  message_norm_min = 0
  message_norm_min_index = 0

  if torch_version_major == 0:  # VERSION
    for d in range(n_labels):
      value = message[dir, h, w, d].clone()  # a reference, must clone this, otherwise the value will be changed
      if d == 0:
        message_norm_min = value
        message_norm_min_index = d
      elif value < message_norm_min:
        message_norm_min = value
        message_norm_min_index = d
  else:
    for d in range(n_labels-1, -1, -1):
      value = message[dir, h, w, d].clone()  # a reference, must clone this, otherwise the value will be changed
      if d == n_labels-1:
        message_norm_min = value
        message_norm_min_index = d
      elif value < message_norm_min:
        message_norm_min = value
        message_norm_min_index = d

  for d in range(n_labels):
    message[dir, h, w, d] -= message_norm_min

  return message, message_norm_min_index


def message_norm_back(dmessage_update, message_norm_indices, iter, dir):
  height, width, n_label = dmessage_update.size()[-3:]

  for h in range(height):
    for w in range(width):
      dmessage_sum = 0
      for d in range(n_label):
        dmessage_sum += dmessage_update[dir,h,w,d]
      dmessage_update[dir,h,w,message_norm_indices[iter,dir,h,w].long()] -= dmessage_sum

  return dmessage_update


def message_norm_one_back(dmessage_update, message_norm_indices, iter, dir, th, tw):
  height, width, n_label = dmessage_update.size()[-3:]

  for h in range(height):
    for w in range(width):
      if (h != th) or (w != tw): continue
      dmessage_sum = 0
      for d in range(n_label):
        dmessage_sum += dmessage_update[dir,h,w,d]
      dmessage_update[dir,h,w,message_norm_indices[iter,dir,h,w].long()] -= dmessage_sum

  return dmessage_update


def message_sum_all(n_dir, dir, disp, h, w, message, message_update):
  message_sum = 0
  for d in range(n_dir):
    if d == dir:
      message_sum += message_update[d,h,w,disp]
    else:
      message_sum += message[d,h,w,disp]
  return message_sum


def message_sum_all_back(n_dir,dir,disp,h,w,value,dmessage,dmessage_update):
  for d in range(n_dir):
    if d == dir:
      dmessage_update[dir,h,w,disp] += value
    else:
      dmessage[d,h,w,disp] += value
  return dmessage, dmessage_update


def get_param(dir, enable_backward=False):
  is_forward_l2r = dir%2==0
  dir_inv = (dir + 1) if is_forward_l2r else (dir - 1)

  if dir == 0:
    h_step, w_step = 0, 1
  elif dir == 1:
    h_step, w_step = 0, -1
  elif dir == 2:
    h_step, w_step = 1, 0
  elif dir == 3:
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

  if enable_backward:
    h_step, w_step = -h_step, -w_step

  return h_step, w_step, dir_inv, is_forward_l2r


def test_mp_module_manual(n_dir, n_iter, unary, label_context_all, rho, enable_backward):
  n_disp, height, width = unary.size()
  unary = unary.permute(1,2,0)
  message = unary.new_zeros(n_dir, height, width, n_disp)
  message_min_indices = unary.new_zeros(n_iter, n_dir, height, width, n_disp)
  message_norm_indices = unary.new_zeros(n_iter, n_dir, height, width)

  # ==== Forward
  for iter in range(n_iter):
    message_update = message.clone()

    # horizontal
    if n_dir >= 2:
      for dir in [0, 1]:
        h_step, w_step, dir_inv, is_forward_l2r = get_param(dir)

        if is_forward_l2r:
          w_start, w_stop = 1, width
        else:
          w_start, w_stop = width-2, -1

        w_move = 1 if (w_start < w_stop) else -1
        label_context = label_context_all[dir,:,:]
        for h in range(height):
          for w in range(w_start,w_stop,w_move):
            for current_d in range(n_disp):
              message_min = 0
              message_min_index = 0

              # Bug 20190803: PyTorch 0.4.1, torch.max and torch.min return min_indices
              # while pytorch 1.1.0 return max_indices, using PyTorch 1.1.0 now

              if torch_version_major == 0:
                for front_d in range(n_disp):
                  front_node_w = w-w_step
                  label_context_value = label_context[front_d,current_d] if is_forward_l2r else label_context[current_d,front_d]
                  message_sum = message_sum_all(n_dir,dir,front_d,h,front_node_w,message,message_update)
                  # From the observation, use unary()
                  value = rho*(unary[h,front_node_w,front_d]+message_sum)-message[dir_inv,h,front_node_w,front_d]+label_context_value

                  if front_d == 0:
                    message_min = value
                    message_min_index = front_d
                  elif value < message_min:
                    message_min = value
                    message_min_index = front_d
              else:
                for front_d in range(n_disp-1, -1, -1):
                  front_node_w = w-w_step
                  label_context_value = label_context[front_d,current_d] if is_forward_l2r else label_context[current_d,front_d]
                  message_sum = message_sum_all(n_dir,dir,front_d,h,front_node_w,message,message_update)
                  # From the observation, use unary()
                  value = rho*(unary[h,front_node_w,front_d]+message_sum)-message[dir_inv,h,front_node_w,front_d]+label_context_value

                  if front_d == n_disp-1:
                    message_min = value
                    message_min_index = front_d
                  elif value < message_min:
                    message_min = value
                    message_min_index = front_d

              message_update[dir,h,w,current_d] = message_min
              message_min_indices[iter,dir,h,w,current_d] = message_min_index

            message_update, message_norm_indices[iter,dir,h,w] = message_norm(message_update, loc=(h, w), dir=dir)
            torch.cuda.empty_cache()

    # vertical
    if n_dir >= 4:
      for dir in [2, 3]:
        h_step, w_step, dir_inv, is_forward_l2r = get_param(dir)

        if is_forward_l2r:
          h_start, h_stop = 1, height
        else:
          h_start, h_stop = height-2, -1

        h_move = 1 if (h_start < h_stop) else -1
        label_context = label_context_all[dir, :, :]
        for w in range(width):
          for h in range(h_start,h_stop,h_move):
            for current_d in range(n_disp):
              message_min = 0
              message_min_index = 0

              if torch_version_major == 0:
                for front_d in range(n_disp):
                  front_node_h = h - h_step
                  label_context_value = label_context[front_d,current_d] if is_forward_l2r else label_context[current_d,front_d]
                  message_sum = message_sum_all(n_dir,dir,front_d,front_node_h,w,message,message_update)
                  value = rho*(unary[front_node_h,w,front_d]+message_sum)-message[dir_inv,front_node_h,w,front_d]+label_context_value

                  if front_d == 0:
                    message_min = value
                    message_min_index = front_d
                  elif value < message_min:
                    message_min = value
                    message_min_index = front_d
              else:
                for front_d in range(n_disp-1, -1, -1):
                  front_node_h = h - h_step
                  label_context_value = label_context[front_d,current_d] if is_forward_l2r else label_context[current_d,front_d]
                  message_sum = message_sum_all(n_dir,dir,front_d,front_node_h,w,message,message_update)
                  value = rho*(unary[front_node_h,w,front_d]+message_sum)-message[dir_inv,front_node_h,w,front_d]+label_context_value

                  if front_d == n_disp-1:
                    message_min = value
                    message_min_index = front_d
                  elif value < message_min:
                    message_min = value
                    message_min_index = front_d

              message_update[dir,h,w,current_d] = message_min
              message_min_indices[iter,dir,h,w,current_d] = message_min_index

            message_update, message_norm_indices[iter,dir,h,w] = message_norm(message_update, loc=(h, w), dir=dir)
            torch.cuda.empty_cache()

    # diagonal
    if n_dir >= 5:
      for dir in range(4,n_dir):
        h_step, w_step, dir_inv, is_forward_l2r = get_param(dir)

        if is_forward_l2r:
          h_start, h_stop = 0, height
        else:
          h_start, h_stop = height-1, -1

        h_move = 1 if (h_start < h_stop) else -1
        label_context = label_context_all[dir, :, :]
        for h in range(h_start,h_stop,h_move):
          for w in range(width):
            front_node_h, front_node_w = h - h_step, w - w_step

            if (front_node_h < 0) or (front_node_h >= height) or (front_node_w < 0) or (front_node_w >= width):
              continue

            for current_d in range(n_disp):
              message_min = 0
              message_min_index = 0

              if torch_version_major == 0:
                for front_d in range(n_disp):
                  label_context_value = label_context[front_d,current_d] if is_forward_l2r else label_context[current_d,front_d]
                  message_sum = message_sum_all(n_dir, dir, front_d, front_node_h, front_node_w, message, message_update)
                  value = rho*(unary[front_node_h,front_node_w,front_d]+message_sum) \
                          -message[dir_inv,front_node_h,front_node_w,front_d]+label_context_value

                  if front_d == 0:
                    message_min = value
                    message_min_index = front_d
                  elif value < message_min:
                    message_min = value
                    message_min_index = front_d
              else:
                for front_d in range(n_disp-1, -1, -1):
                  label_context_value = label_context[front_d,current_d] if is_forward_l2r else label_context[current_d,front_d]
                  message_sum = message_sum_all(n_dir, dir, front_d, front_node_h, front_node_w, message, message_update)
                  value = rho*(unary[front_node_h,front_node_w,front_d]+message_sum) \
                          -message[dir_inv,front_node_h,front_node_w,front_d]+label_context_value

                  if front_d == n_disp-1:
                    message_min = value
                    message_min_index = front_d
                  elif value < message_min:
                    message_min = value
                    message_min_index = front_d

              message_update[dir,h,w,current_d] = message_min
              message_min_indices[iter,dir,h,w,current_d] = message_min_index

            message_update, message_norm_indices[iter,dir,h,w] = message_norm(message_update, loc=(h, w), dir=dir)
            torch.cuda.empty_cache()

    message = message_update.clone()

  # Fake loss grad
  cost_final = -(unary + message_update.sum(0))

  if enable_backward:
    dunary = -message.new_ones(unary.size())
    dmessage_update = -message.new_ones(message_update.size())
    dlabel_context_all = message.new_zeros(n_dir, n_disp, n_disp)

    # ==== Backward
    for iter in range(n_iter-1,-1,-1):
      dmessage = message.new_zeros(message.size())

      # diagonal
      if n_dir > 4:
        for dir in range(n_dir-1,3,-1):
          h_step, w_step, dir_inv, is_forward_l2r = get_param(dir, enable_backward=True)

          if is_forward_l2r:
            h_start, h_stop = height-1, -1
          else:
            h_start, h_stop = 0, height

          dmessage_update = message_norm_back(dmessage_update,message_norm_indices,iter=iter,dir=dir)
          torch.cuda.empty_cache()

          # The front node is in accordance with forward pass
          h_move = 1 if (h_start < h_stop) else -1
          dlabel_context = dlabel_context_all[dir,:,:]
          for h in range(h_start,h_stop,h_move):
            for w in range(width):
              front_node_h, front_node_w = h + h_step, w + w_step
              if (front_node_h < 0) or (front_node_h >= height) or (front_node_w < 0) or (front_node_w >= width):
                continue
              for current_d in range(n_disp):
                front_d = message_min_indices[iter,dir,h,w,current_d].long()
                value = dmessage_update[dir,h,w,current_d]
                dunary[front_node_h,front_node_w,front_d] += value*rho
                dmessage, dmessage_update = message_sum_all_back(n_dir,dir,front_d,front_node_h,front_node_w,
                                                                 value*rho,dmessage,dmessage_update)
                dmessage[dir_inv,front_node_h,front_node_w,front_d] -= value
                if is_forward_l2r:
                  dlabel_context[front_d,current_d] += value
                else:
                  dlabel_context[current_d,front_d] += value
          dmessage_update[dir, :, :, :] = 0

      # vertical
      if n_dir >= 4:
        for dir in [3, 2]:
          h_step, w_step, dir_inv, is_forward_l2r = get_param(dir, enable_backward=True)

          if is_forward_l2r:
            h_start, h_stop = height-1, 0
          else:
            h_start, h_stop = 0, height-1

          dmessage_update = message_norm_back(dmessage_update,message_norm_indices,iter=iter,dir=dir)
          torch.cuda.empty_cache()
          h_move = 1 if (h_start < h_stop) else -1
          dlabel_context = dlabel_context_all[dir, :, :]
          for w in range(width):
            for h in range(h_start,h_stop,h_move):
              for current_d in range(n_disp):
                front_node_h = h + h_step
                front_d = message_min_indices[iter,dir,h,w,current_d].long()
                value = dmessage_update[dir,h,w,current_d]
                dunary[front_node_h,w,front_d] += value*rho
                dmessage, dmessage_update = message_sum_all_back(n_dir,dir,front_d,front_node_h,
                                                                 w,value*rho,dmessage,dmessage_update)
                dmessage[dir_inv,front_node_h,w,front_d] -= value
                if is_forward_l2r:
                  dlabel_context[front_d,current_d] += value
                else:
                  dlabel_context[current_d,front_d] += value
          dmessage_update[dir, :, :, :] = 0

      # horizontal
      if n_dir >= 2:
        for dir in [1, 0]:
          h_step, w_step, dir_inv, is_forward_l2r = get_param(dir, enable_backward=True)

          if is_forward_l2r:
            w_start, w_stop = width-1, 0
          else:
            w_start, w_stop = 0, width-1

          dmessage_update = message_norm_back(dmessage_update,message_norm_indices,iter=iter,dir=dir)
          torch.cuda.empty_cache()
          w_move = 1 if (w_start < w_stop) else -1
          dlabel_context = dlabel_context_all[dir, :, :]
          for h in range(height):
            for w in range(w_start,w_stop,w_move):
              for current_d in range(n_disp):
                front_node_w = w + w_step
                front_d = message_min_indices[iter,dir,h,w,current_d].long()
                value = dmessage_update[dir,h,w,current_d]
                dunary[h,front_node_w,front_d] += value*rho
                dmessage, dmessage_update = message_sum_all_back(n_dir,dir,front_d,h,front_node_w,
                                                                 value*rho,dmessage,dmessage_update)
                dmessage[dir_inv,h,front_node_w,front_d] -= value
                if is_forward_l2r:
                  dlabel_context[front_d,current_d] += value
                else:
                  dlabel_context[current_d,front_d] += value
          dmessage_update[dir, :, :, :] = 0

      dmessage += dmessage_update
      dmessage_update = dmessage.clone()

    return cost_final.permute(2,0,1), message_update, message_min_indices, message_norm_indices, \
           dunary.permute(2,0,1), dmessage, dlabel_context_all
  else:
    return cost_final.permute(2,0,1), message_update, message_min_indices, message_norm_indices, None, None, None
