from modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, enable_dff=False, pretrained_path=None):
    if backbone in {'resnet50', 'resnet101'}:
        return resnet.ResNetSerial(backbone, output_stride, BatchNorm, enable_dff=enable_dff,
                                   pretrained_path=pretrained_path)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
