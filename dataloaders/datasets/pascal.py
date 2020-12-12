from __future__ import print_function, division
import os, torch, sys
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from utils.mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from skimage import feature, filters
from skimage.transform import rescale
import torch.nn.functional as F
sys.path.append('../..')
from mpnet import getEdgeShift


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 server='039614',
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        if args.dataset_root is None:
            base_dir = Path.db_root_dir('pascal', server=server)
        else:
            base_dir = args.dataset_root

        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')

        if args.edge_mode == 'superpixel_edge':
            if args.superpixel == 'CRISP':
                self._edge_dir = os.path.join(self._base_dir, 'SuperPixel/CRISP/{:.2f}'.format(args.superpixel_threshold))
            elif args.superpixel == 'BASS':
                self._edge_dir = os.path.join(self._base_dir, 'SuperPixel/BASS/{:d}'.format(int(args.superpixel_threshold)))
            else:
                assert False
        else:
            self._edge_dir = os.path.join(self._base_dir, 'BerkeleyEdges/{}'.format(args.edge_pixel))

        if args.mode == 'weakly':
            self._cat_dir = os.path.join(self._base_dir, 'pascal_2012_scribble')
        else:
            if args.edge_mode in ['edge_net', 'edge_net_sigmoid']:
                # self._cat_dir = os.path.join(self._base_dir, 'SegmentationAugAccEdges')
                self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
            else:
                self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        if self.args.use_small:
            _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'SegmentationSub')
        else:
            if self.args.edge_mode in ['edge_net', 'edge_net_sigmoid']:
                # _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Berkeley_exclude_VOC_val')
                _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'SegmentationAug')
            else:
                _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'SegmentationAug')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.image_names = []
        self.edges = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.image_names.append(line)

                if args.edge_mode == 'superpixel_edge':
                    _edge = os.path.join(self._edge_dir, line + ".png")
                    assert os.path.isfile(_edge)
                    self.edges.append(_edge)
                else:
                    self.edges.append(None)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def _make_img_gt_point_pair(self, index):
        # Zhiwei
        # image_path = '/home/xu064/WorkSpace/git-lab/pytorch-projects/rloss/data/pascal_scribble/JPEGImages/2007_000129.jpg'
        # category_path = '/home/xu064/WorkSpace/git-lab/pytorch-projects/rloss/data/pascal_scribble/SegmentationClassAug/2007_000129.png'
        # edge_path = self.edges[index]

        image_path = self.images[index]
        category_path = self.categories[index]
        edge_path = self.edges[index]

        _img = Image.open(image_path).convert('RGB')
        _target = Image.open(category_path)
        _edge = Image.open(edge_path) if (edge_path is not None) else None

        return _img, _target, _edge

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target, _edge = self._make_img_gt_point_pair(index)
        _padding_mask = Image.fromarray(np.ones((_img.height, _img.width), dtype=np.uint8))
        sample = {'image': _img, 'label': _target, 'padding_mask': _padding_mask,
                  'size': (_img.height, _img.width)}

        if (self.args.edge_mode in ['edge_net', 'superpixel_edge']) and (_edge is not None):
            # edge:(1,h,w)
            sample.update({'edge': _edge})

        for split in self.split:
            if split == "train":
                sample = self.transform_tr_part1_1(sample)

                if self.args.edge_mode == 'canny':
                    canny = self.run_canny(sample['image'])
                    sample.update({'edge': canny})
                elif self.args.edge_mode == 'sobel':
                    sobel = self.run_sobel(sample['image'])
                    sample.update({'edge': sobel})

                sample = self.transform_tr_part1_2(sample)
            elif split == 'val':
                sample = self.transform_val_part1(sample)

                if self.args.edge_mode == 'canny':
                    canny = self.run_canny(sample['image'])
                    sample.update({'edge': canny})
                elif self.args.edge_mode == 'sobel':
                    sobel = self.run_sobel(sample['image'])
                    sample.update({'edge': sobel})
            else:
                assert False

            if self.args.mpnet_edge_weight_fn is not None:
                padding_mask_tensor = torch.from_numpy(np.array(sample['padding_mask'])).float().unsqueeze(0)

                if self.args.edge_mode in ['gt_edge', 'superpixel_edge', 'kernel_cue_real',
                                           'kernel_cue_binary', 'threshold']:
                    # img_for_edge:(c,h,w)
                    if self.args.edge_mode == 'gt_edge':
                        img_for_edge = torch.from_numpy(np.array(sample['label'])).float().unsqueeze(0)
                    elif self.args.edge_mode == 'superpixel_edge':
                        img_for_edge = torch.from_numpy(np.array(sample['edge'])).float().permute(2, 0, 1)
                    else:
                        img_for_edge = torch.from_numpy(np.array(sample['image'])).float().permute(2, 0, 1)

                    # Ensure padded parts are filled with 0 penalty
                    # edge_weights:(ndir,h,w)
                    edge_weights = self.args.mpnet_edge_weight_fn(self.args.edge_mode,
                                                                  img_for_edge,
                                                                  self.args.mpnet_n_dirs,
                                                                  self.args.mpnet_scale_list,
                                                                  sigma=self.args.mpnet_sigma,
                                                                  padding_mask=padding_mask_tensor)
                    sample.update({'edge_weights': edge_weights})
                elif self.args.edge_mode in {'canny', 'sobel'}:
                    edge2shift = torch.from_numpy(sample['edge'])
                    edge_weights = getEdgeShift(self.args.edge_mode,
                                                edge2shift,
                                                self.args.mpnet_n_dirs)

                    # Cropping mask
                    padding_mask_tensor = torch.from_numpy(np.array(sample['padding_mask'])).float().unsqueeze(0)

                    if padding_mask_tensor.size()[-2:] != edge_weights.size()[-2:]:
                        padding_mask_tensor = F.interpolate(padding_mask_tensor.unsqueeze(0),
                                                            size=(edge_weights.size()[-2:]),
                                                            mode='nearest')
                        padding_mask_tensor = padding_mask_tensor.squeeze(0)

                    edge_weights *= padding_mask_tensor
                    sample.update({'edge_weights': [edge_weights]})

            if split == 'train':
                sample = self.transform_tr_part2(sample)
            elif split == 'val':
                sample = self.transform_val_part2(sample)

        sample.update({'name': self.image_names[index]})

        if 'padding_mask' in sample:
            del sample['padding_mask']

        return sample

    def transform_tr_part1(self, sample):
        if self.args.use_small:
            composed_transforms = transforms.Compose([
                tr.FixScaleCrop(crop_size=self.args.crop_size)])
        else:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
                tr.RandomGaussianBlur()])  # Zhiwei

        return composed_transforms(sample)

    def transform_tr_part1_1(self, sample):
        if self.args.use_small:
            composed_transforms = transforms.Compose([
                tr.FixScaleCrop(crop_size=self.args.crop_size)])
        else:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size)])  # Zhiwei

        return composed_transforms(sample)

    def transform_tr_part1_2(self, sample):
        if not self.args.use_small:
            composed_transforms = transforms.Compose([tr.RandomGaussianBlur()])

        return composed_transforms(sample)

    def transform_tr_part2(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val_part1(self, sample):
        if self.args.enable_test and self.args.enable_test_full:
            return sample
        else:
            if self.args.enable_adjust_val:
                composed_transforms = transforms.Compose([
                    tr.AutoAdjustSize(factor=self.args.adjust_val_factor, fill=254)])
            else:
                composed_transforms = transforms.Compose([
                    tr.FixScaleCrop(crop_size=self.args.crop_size)])

            return composed_transforms(sample)

    def transform_val_part2(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

    def run_canny(self, image, mask=None):
        image = np.array(image.convert('L'))
        image = rescale(image, self.args.mpnet_scale_list[0], anti_aliasing=True)
        edge = feature.canny(image, sigma=self.args.mpnet_scale_list[0])

        return edge.astype(np.float32)

    def run_sobel(self, image, mask=None):
        image = np.array(image.convert('L'))
        image = rescale(image, self.args.mpnet_scale_list[0], anti_aliasing=True)
        edge = filters.sobel(image)
        edge = edge / edge.max()

        return edge.astype(np.float32)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


