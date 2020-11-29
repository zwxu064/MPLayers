import os
import time
import torch
import torch.nn as nn
import copy
import sys
sys.path.append('..')
from lib import TRWP


def softmax_forward(softmax_dim, data):
  prob = nn.Softmax(dim=softmax_dim)(-data)
  return prob


def softmax_backward(softmax_dim, prob, dprob):
  return -prob * (dprob - (prob * dprob).sum(softmax_dim, keepdim=True))


def softmax_message(message, softmax_dim, enable_softmax_inbuilt=False):
  if not enable_softmax_inbuilt:
    # use min not max since input to exp is -message
    message_min, _ = message.min(softmax_dim, keepdim=True)
    message_norm = message - message_min
    # message_norm = message
    msg_exp = torch.exp(-message_norm)
    prob = msg_exp / msg_exp.sum(dim=softmax_dim, keepdim=True)
    soft_weighted_message = message * prob
  else:
    prob = nn.Softmax(dim=softmax_dim)(-message)
    soft_weighted_message = prob * message

  return soft_weighted_message, prob


def softmax_message_grad(softmax_dim, soft_weighted_message, prob, dsoft_weighted_message):
  dmessage = dsoft_weighted_message * (prob - soft_weighted_message +
                                       prob * soft_weighted_message.sum(softmax_dim, keepdim=True))
  return dmessage


if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  torch.manual_seed(2019)
  torch.cuda.manual_seed_all(2019)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  enable_backward = True

  n_labels = 21
  message = torch.rand((1024, n_labels, n_labels)).float().cuda()
  message.requires_grad = enable_backward
  softmax_dim = 1

  print('Start unittest of soft-weighted belief propagation.')

  # ==== Exp 1
  # Option 1
  message_inbuilt = copy.deepcopy(message)
  torch.cuda.empty_cache()
  torch.cuda.synchronize()
  time_start = time.time()
  soft_weighted_message_inbuilt, _ = softmax_message(message_inbuilt, softmax_dim=softmax_dim,
                                                     enable_softmax_inbuilt=True)
  soft_weighted_message_inbuilt_sum = soft_weighted_message_inbuilt.sum(softmax_dim, keepdim=True)

  if enable_backward:
    loss = soft_weighted_message_inbuilt_sum.sum()
    loss.backward()
    torch.cuda.synchronize()
    duration = time.time() - time_start
    print('Built:', soft_weighted_message_inbuilt_sum.detach().sum(), message_inbuilt.grad.sum(), duration)
  else:
    torch.cuda.synchronize()
    duration = time.time() - time_start
    print('Built:', soft_weighted_message_inbuilt_sum.detach().sum(), duration)

  # Option 2
  message = copy.deepcopy(message)
  torch.cuda.empty_cache()
  torch.cuda.synchronize()
  time_start = time.time()
  soft_weighted_message, prob = softmax_message(message, softmax_dim=softmax_dim,
                                                enable_softmax_inbuilt=False)
  soft_weighted_message_sum = soft_weighted_message.sum(softmax_dim, keepdim=True)

  if enable_backward:
    loss = soft_weighted_message_sum.sum()
    loss.backward()
    torch.cuda.synchronize()
    duration = time.time() - time_start
    print('Non-inbuilt:', soft_weighted_message_sum.detach().sum(), message.grad.sum(), duration)
  else:
    torch.cuda.synchronize()
    duration = time.time() - time_start
    print('Non-inbuilt:', soft_weighted_message_sum.detach().sum(), duration)

  # Option 3: manual grad
  dsoft_weighted_message = soft_weighted_message.new_ones(soft_weighted_message.size())

  if enable_backward:
    dmessage = softmax_message_grad(softmax_dim, soft_weighted_message.detach(), prob.detach(), dsoft_weighted_message)
    print('Manual:', soft_weighted_message_sum.detach().sum(), dmessage.sum())
  else:
    print('Manual:', soft_weighted_message_sum.detach().sum())

  # Compare
  print('Softweighted message gap:', (soft_weighted_message_inbuilt - soft_weighted_message).detach().abs().max())

  if enable_backward:
    print('Softweighted message grad gap, inbuilt vs non-inbuilt:', (message_inbuilt.grad - message.grad).abs().max())
    print('Softweighted message grad gap, inbuilt vs manual:', (message_inbuilt.grad - dmessage).abs().max())

  # ==== Exp 2
  # Test CUDA softmax
  message_cuda = copy.deepcopy(message)

  # Forward
  message_cuda_soft_sum_norm = message_cuda.new_zeros((message_cuda.size(0), message_cuda.size(2)))
  message_cuda_soft_sum_min_ind = message_cuda.new_zeros((message.size(0), 1), dtype=torch.uint8)

  for i in range(10):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time_start = time.time()
    TRWP.soft_weighted_message(message_cuda, message_cuda_soft_sum_norm, message_cuda_soft_sum_min_ind)
    torch.cuda.synchronize()
    duration = time.time() - time_start

    # message_min, _ = message_cuda.min(softmax_dim, keepdim=True)
    # message_norm = message_cuda - message_min
    message_cuda.requires_grad = True
    message_prob = nn.Softmax(softmax_dim)(-message_cuda)
    message_soft = message_prob * message_cuda
    message_soft_sum = message_soft.sum(softmax_dim)
    message_soft_sum_min, message_soft_sum_min_ind = message_soft_sum.min(softmax_dim, keepdim=True)
    message_soft_sum_norm = message_soft_sum - message_soft_sum_min
    print('CUDA GAP: msg soft sum: {}, min index: {}, time: {:.8f}' \
          .format((message_soft_sum_norm - message_cuda_soft_sum_norm).abs().max(),
                  (message_soft_sum_min_ind.float() - message_cuda_soft_sum_min_ind.float()).abs().max(),
                  duration))

  # Backpropagation
  loss = 0.666 * message_soft_sum_norm.sum()
  loss.backward()

  dmessage_cuda_soft_sum_norm = 0.666 * message_cuda_soft_sum_norm.new_ones(message_cuda_soft_sum_norm.size())
  dmessage_cuda = message_cuda.new_zeros(message_cuda.size())

  for i in range(10):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time_start = time.time()
    TRWP.soft_weighted_message_back(dmessage_cuda_soft_sum_norm, message_cuda, message_cuda_soft_sum_min_ind, dmessage_cuda)
    torch.cuda.synchronize()
    duration = time.time() - time_start

    print('CUDA GAP back: dmessage: {}, time: {}' \
          .format((message_cuda.grad - dmessage_cuda).abs().max(), duration))

  # ==== Test softmax
  data = copy.deepcopy(message)
  data.requires_grad = True
  prob = softmax_forward(softmax_dim, data)
  loss = prob.sum()
  loss.backward()

  dprob = prob.new_ones(prob.size())
  ddata = softmax_backward(softmax_dim, prob.detach(), dprob)
  print('Test softmax gap:', (data.grad - ddata).abs().max())
