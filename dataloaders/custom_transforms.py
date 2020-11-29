import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        edge = sample['edge'] if ('edge' in sample) else None
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        edge = np.array(edge).astype(np.float32) if (edge is not None) else None
        img /= 255.0
        img -= self.mean
        img /= self.std

        sample['image'] = img
        sample['label'] = mask

        if edge is not None:
            sample['edge'] = edge

        return sample


class NormalizeImage(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        edge = sample['edge'] if ('edge' in sample) else None
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        edge = np.array(edge).astype(np.float32) if (edge is not None) else None

        sample['image'] = torch.from_numpy(img).float()
        sample['label'] = torch.from_numpy(mask).float()

        if edge is not None:
            sample['edge'] = torch.from_numpy(edge).float()

        return sample


class ToTensorImage(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        padding_mask = sample['padding_mask'] if ('padding_mask' in sample) else None
        edge = sample['edge'] if ('edge' in sample) else None

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            padding_mask = padding_mask.transpose(Image.FLIP_LEFT_RIGHT) if (padding_mask is not None) else None
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT) if (edge is not None) else None

        sample['image'] = img
        sample['label'] = mask

        if padding_mask is not None:
            sample['padding_mask'] = padding_mask

        if edge is not None:
            sample['edge'] = edge

        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        padding_mask = sample['padding_mask'] if ('padding_mask' in sample) else None
        edge = sample['edge'] if ('edge' in sample) else None

        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        padding_mask = padding_mask.rotate(rotate_degree, Image.NEAREST) if (padding_mask is not None) else None
        edge = edge.rotate(edge, Image.NEAREST) if (edge is not None) else None

        sample['image'] = img
        sample['label'] = mask

        if padding_mask is not None:
            sample['padding_mask'] = padding_mask

        if edge is not None:
            sample['edge'] = edge

        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=254):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        padding_mask = sample['padding_mask'] if ('padding_mask' in sample) else None
        edge = sample['edge'] if ('edge' in sample) else None

        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        padding_mask = padding_mask.resize((ow, oh), Image.NEAREST) if (padding_mask is not None) else None
        edge = edge.resize((ow, oh), Image.NEAREST) if (edge is not None) else None

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
            padding_mask = ImageOps.expand(padding_mask, border=(0, 0, padw, padh), fill=0) if (padding_mask is not None) else None
            edge = ImageOps.expand(edge, border=(0, 0, padw, padh), fill=0) if (edge is not None) else None

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        padding_mask = padding_mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size)) if (padding_mask is not None) else None
        edge = edge.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size)) if (edge is not None) else None

        sample['image'] = img
        sample['label'] = mask

        if padding_mask is not None:
            sample['padding_mask'] = padding_mask

        if edge is not None:
            sample['edge'] = edge

        return sample


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        padding_mask = sample['padding_mask'] if ('padding_mask' in sample) else None
        edge = sample['edge'] if ('edge' in sample) else None

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        padding_mask = padding_mask.resize((ow, oh), Image.NEAREST) if (padding_mask is not None) else None
        edge = edge.resize((ow, oh), Image.NEAREST) if (edge is not None) else None

        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        padding_mask = padding_mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size)) if (padding_mask is not None) else None
        edge = edge.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size)) if (edge is not None) else None

        sample['image'] = img
        sample['label'] = mask

        if padding_mask is not None:
            sample['padding_mask'] = padding_mask

        if edge is not None:
            sample['edge'] = edge

        return sample


class AutoAdjustSize(object):
    def __init__(self, factor=32, fill=254):
        self.factor = factor
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        padding_mask = sample['padding_mask'] if ('padding_mask' in sample) else None
        edge = sample['edge'] if ('edge' in sample) else None

        w, h = img.size
        oh = ((h + self.factor - 1) // self.factor) * self.factor
        ow = ((w + self.factor - 1) // self.factor) * self.factor

        padh = max(0, oh - h)
        padw = max(0, ow - w)
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        padding_mask = ImageOps.expand(padding_mask, border=(0, 0, padw, padh), fill=0) if (padding_mask is not None) else None
        edge = ImageOps.expand(edge, border=(0, 0, padw, padh), fill=0) if (edge is not None) else None

        sample['image'] = img
        sample['label'] = mask

        if padding_mask is not None:
            sample['padding_mask'] = padding_mask

        if edge is not None:
            sample['edge'] = edge

        return sample


class FixScaleCropImage(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        padding_mask = sample['padding_mask'] if ('padding_mask' in sample) else None
        edge = sample['edge'] if ('edge' in sample) else None

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        padding_mask = padding_mask.resize(self.size, Image.NEAREST) if (padding_mask is not None) else None
        edge = edge.resize(self.size, Image.NEAREST) if (edge is not None) else None

        sample['image'] = img
        sample['label'] = mask

        if padding_mask is not None:
            sample['padding_mask'] = padding_mask

        if edge is not None:
            sample['edge'] = edge

        return sample


def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *=255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))

    return torch.tensor(images)
