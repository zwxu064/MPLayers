import math
import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from utils import init_params


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, model_name, block, layers, output_stride, BatchNorm, pretrained=True,
                 enable_dff=False, pretrained_path=None):
        self.model_name = model_name
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.enable_dff = enable_dff
        blocks = [1, 2, 4] if (not enable_dff) else [1, 1, 1]  # rloss: [1,2,4]; dff: [1,1,1]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2] if (not enable_dff) else [2, 2, 2, 4]  # rloss: [1,1,1,2]; dff: [2,2,2,4] Zhiwei
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules Zhiwei
        if enable_dff:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pretrained_path = pretrained_path

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        init_params(self)
        # self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []

        # Zhiwei rloss: blocks[0] * dilation; dff: dilation
        adj_dilation = blocks[0] * dilation
        layers.append(block(self.inplanes, planes, stride, dilation=adj_dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            # Zhiwei rloss: blocks[i] * dilation; dff: dilation
            adj_dilation = blocks[i] * dilation
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=adj_dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        low_level_feat = []
        feat_dff = []

        x = self.conv1(input)
        x = self.bn1(x)
        x_0 = self.relu(x)
        x = self.maxpool(x_0)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        # Zhiwei
        if self.enable_dff:
            feat_dff.append(x_0)
            feat_dff.append(x_1)
            feat_dff.append(x_2)
            feat_dff.append(x_3)
            feat_dff.append(x_4)

        low_level_feat.append(x_0)
        low_level_feat.append(x_1)  # For decoder
        # low_level_feat.append(x_2)
        # low_level_feat.append(x_3)
        # low_level_feat.append(x_4)

        return x_4, low_level_feat, feat_dff

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
                nn.init.kaiming_normal_(m.bias) if (m.bias is not None) else None
            elif isinstance(m, SynchronizedBatchNorm2d) \
                    or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_dict(self):
        if self.model_name == 'resnet101':
            pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        elif self.model_name == 'resnet50':
            pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')

        return pretrain_dict

    def _load_pretrained_model(self):
        if self.pretrained_path is not None:
            if os.path.exists(self.pretrained_path):
                pretrain_dict = torch.load(self.pretrained_path)
            else:
                print('!!! Warning, {} not exist, use pytorch version.'.format(self.pretrained_path))
                pretrain_dict = self._load_pretrained_dict()
        else:
            pretrain_dict = self._load_pretrained_dict()

        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNetSerial(model_name, output_stride, BatchNorm, pretrained=True, enable_dff=False, pretrained_path=None):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if model_name == 'resnet50':
        num_layers = 6
    elif model_name == 'resnet101':
        num_layers = 23
    else:
        assert False

    model = ResNet(model_name, Bottleneck, [3, 4, num_layers, 3], output_stride, BatchNorm, pretrained=pretrained,
                   enable_dff=enable_dff, pretrained_path=pretrained_path)
    return model


if __name__ == "__main__":
    import torch
    model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())