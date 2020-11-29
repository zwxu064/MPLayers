import torch
import torch.nn as nn
import numpy as np


def create_label_context(args):
  enable_seg = True if (args.n_classes == 21) else False

  if enable_seg:
    label_context = np.zeros((args.n_classes, args.n_classes), dtype=np.float32)
    if args.mpnet_smoothness_mode in {'Potts', 'potts'}:
      if args.mpnet_smoothness_train != 'sigmoid':
        for left_d in range(args.n_classes):
          for right_d in range(args.n_classes):
            gap = right_d - left_d
            if gap > 0:
              label_context[left_d, right_d] = 1
    elif args.mpnet_smoothness_mode == 'TL':
      for left_d in range(args.n_classes):
        for right_d in range(args.n_classes):
          gap = right_d - left_d
          if gap > 0:
            label_context[left_d, right_d] = gap
    elif args.mpnet_smoothness_mode == 'TQ':
      for left_d in range(args.n_classes):
        for right_d in range(args.n_classes):
          gap = right_d - left_d
          if gap > 0:
            label_context[left_d, right_d] = np.power(gap, 2)
    elif args.mpnet_smoothness_mode == 'NPotts':
      for left_d in range(args.n_classes):
        for right_d in range(args.n_classes):
          if left_d == right_d:
            label_context[left_d, right_d] = -1

    if (args.mpnet_smoothness_trunct_loc >= 0) and (args.mpnet_smoothness_trunct_value >= 0):
      for left_d in range(args.n_classes):
        for right_d in range(args.n_classes):
          gap = right_d - left_d
          if gap > 0 and gap >= args.mpnet_smoothness_trunct_loc:
            label_context[left_d, right_d] = args.mpnet_smoothness_trunct_value
  else:
    label_context = np.zeros((args.n_classes), dtype=np.float32)
    if args.mpnet_smoothness_mode in {'Potts', 'potts'}:
      if args.mpnet_smoothness_train != 'sigmoid':
        label_context[1:] = 1
    elif args.mpnet_smoothness_mode == 'TL':
      for d in range(args.n_classes):
        label_context[d] = d
    elif args.mpnet_smoothness_mode == 'TQ':
      for d in range(args.n_classes):
        label_context[d] = np.power(d, 2)
    elif args.mpnet_smoothness_mode == 'NPotts':
      label_context[0] = -1

    if (args.mpnet_smoothness_trunct_loc >= 0) and (args.mpnet_smoothness_trunct_value >= 0):
      for d in range(args.n_classes):
        if d >= args.mpnet_smoothness_trunct_loc:
          label_context[d] = args.mpnet_smoothness_trunct_value

  label_context = torch.from_numpy(label_context)
  label_context = label_context * args.mpnet_term_weight

  if args.enable_cuda and (not label_context.is_cuda):
    label_context = label_context.cuda()

  enable_grad = args.mpnet_smoothness_train in {'softmax', 'sigmoid', 'on'}
  label_context = nn.Parameter(label_context, requires_grad=enable_grad)

  if len(label_context.size()) == 2:
    label_context_loc = np.triu_indices(args.n_classes, k=1)
    label_context_diag_loc = [range(args.n_classes), range(args.n_classes)]
  else:
    label_context_loc = range(1, args.n_classes)
    label_context_diag_loc = range(1)
  
  return label_context, label_context_loc, label_context_diag_loc