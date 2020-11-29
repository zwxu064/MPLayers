import torch
import os
from MP_module import MPModule

if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = "3"
  enable_cuda = torch.cuda.is_available()
  device = torch.device('cuda' if enable_cuda else 'cpu')
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  enable_backward = True

  mode, enable_soft = 'TRWP', True
  repeats_cuda, n_iter, n_dir, n_disp, h, w, n_cv = 1, 5, 4, 21, 128, 128, 1
  rho = 0.5 if (mode == 'TRWP') else 1
  enable_my_cuda = True
  batch = 12

  assert n_disp <= 64
  seed = 2019

  # ==== random data =================
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  unary = torch.randint(0, 255, (n_cv, n_disp, h, w), dtype=torch.float32, device=device, requires_grad=True)
  label_context = torch.randint(0, n_disp, (n_dir, n_disp, n_disp), dtype=torch.float32, device=device, requires_grad=True)
  edge_weight = torch.randint(1, 5, (n_dir, h, w), dtype=torch.float32, device=device, requires_grad=True)

  unary = unary.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
  edge_weight = edge_weight.unsqueeze(0).repeat(batch, 1, 1, 1)

  mp_module = MPModule(n_dir=n_dir,
                       n_iter=n_iter,
                       n_disp=n_disp,
                       mode=mode,
                       rho=rho,
                       enable_cuda=enable_my_cuda,
                       label_context=label_context,
                       smoothness_train='softmax',
                       enable_soft=enable_soft)

  if enable_backward:
    mp_module.train()
  else:
    mp_module.eval()

  cost_final6, label_context6, _, message_vector_cuda, message_index_cuda = \
    mp_module(unary, edge_weight)

  cost_final6 = -cost_final6  # convert to fake prob

  if enable_backward:
    loss = cost_final6.sum()
    mp_module.label_context.retain_grad()
    unary.retain_grad()
    edge_weight.retain_grad()
    loss.backward()


  for batch_idx in range(batch):
    print('batch:', batch_idx, cost_final6[batch_idx].sum().item(), unary.grad[batch_idx].sum().item(),
          edge_weight.grad[batch_idx].sum().item())