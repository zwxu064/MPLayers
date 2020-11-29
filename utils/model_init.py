import torch.nn as nn
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def init_params(models, is_edgenet=False):
  if not isinstance(models, list):
    models = [models]

  for model in models:
    for m in model.modules():
      if isinstance(m, nn.Conv2d) \
              or isinstance(m, nn.Linear) \
              or isinstance(m, nn.ConvTranspose2d):
        if is_edgenet:
          nn.init.normal_(m.weight, mean=0, std=1e-6)
        else:
          nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0) if (m.bias is not None) else None
      elif isinstance(m, SynchronizedBatchNorm2d) \
              or isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)