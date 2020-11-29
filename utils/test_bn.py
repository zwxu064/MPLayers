import torch
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.sync_batchnorm.replicate import patch_replication_callback


class mymodel(torch.nn.Module):
  def __init__(self, n_class, enable_sync_bn=False):
    super(mymodel, self).__init__()

    if enable_sync_bn:
      self.bn = SynchronizedBatchNorm2d(n_class, affine=False, eps=0)
    else:
      self.bn = torch.nn.BatchNorm2d(n_class, affine=False, eps=0, track_running_stats=False, momentum=1)

  def forward(self, x):
    return self.bn(x)


if __name__ == '__main__':
  import os
  os.environ['CUDA_VISIBLE_DEVICES']="0,3"

  torch.manual_seed(2019)
  torch.cuda.manual_seed(2019)
  c = 3
  # input = torch.rand((12, c, 512, 512), dtype=torch.float32)
  input = torch.rand((12, c, 512, 512), dtype=torch.float32) * 100
  model = mymodel(c, enable_sync_bn=False)
  model = model.cuda()

  input = input.cuda()
  result1 = model(input)


  model = mymodel(c, enable_sync_bn=True)
  model = torch.nn.DataParallel(model, device_ids=[0,1])
  patch_replication_callback(model)
  model = model.cuda()

  input = input.cuda()
  result2 = model(input)

  print((result1 - result2).abs().max(), result1.abs().double().sum(), result2.abs().double().sum())