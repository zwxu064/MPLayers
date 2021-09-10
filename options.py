import argparse
import random
import numpy as np
import torch
import os


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)  # affect randomCrop
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  os.environ['PYTHONHASHSEED']=str(seed)


def set_config():
  parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
  parser.add_argument('--server', type=str, default='039614', choices=['039614', 'data61'], help='039614 or data61')
  parser.add_argument('--backbone', type=str, default='resnet101',
                      choices=['resnet50', 'resnet101', 'xception', 'drn', 'mobilenet'],
                      help='backbone name (default: resnet)')
  parser.add_argument('--out-stride', type=int, default=16, help='network output stride (default: 8)')
  parser.add_argument('--dataset_root', type=str, default=None)
  parser.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'coco', 'cityscapes'],
                      help='dataset name (default: pascal)')
  parser.add_argument('--use-sbd', action='store_true', default=False, help='whether to use SBD dataset (default: True)')
  parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
  parser.add_argument('--base-size', type=int, default=512, help='base image size')
  parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
  parser.add_argument('--freeze-bn', action='store_true', default=False,
                      help='whether to freeze bn parameters (default: False)')
  parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'], help='loss func type (default: ce)')
  # training hyper params
  parser.add_argument('--epochs', type=int, default=None, metavar='N', help='number of epochs to train (default: auto)')
  parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
  parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                      help='input batch size for training (default: auto)')
  parser.add_argument('--val-batch-size', type=int, default=None, metavar='N',
                      help='input batch size for training (default: auto)')
  parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                      help='whether to use balanced weights (default: False)')
  # optimizer params
  parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (default: auto)')
  parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos', 'fixed'],
                      help='lr scheduler mode: (default: poly)')
  parser.add_argument('--warmup_epochs', type=float, default=0)
  parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
  parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
  parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')
  # cuda, seed and logging
  parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
  parser.add_argument('--gpu-ids', type=str, default='0',
                      help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
  parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
  # checking point
  parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
  parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
  # finetuning pre-trained models
  parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset')
  # evaluation option
  parser.add_argument('--eval-interval', type=int, default=1, help='evaluuation interval (default: 1)')
  parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')
  # model saving option
  parser.add_argument('--save-interval', type=int, default=None, help='save model interval in epochs')

  # rloss options
  parser.add_argument('--densecrfloss', type=float, default=0, metavar='M', help='densecrf loss (default: 0)')
  parser.add_argument('--rloss-scale', type=float, default=1.0,
                      help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
  parser.add_argument('--sigma-rgb', type=float, default=15.0, help='DenseCRF sigma_rgb')
  parser.add_argument('--sigma-xy', type=float, default=100.0, help='DenseCRF sigma_xy')
  parser.add_argument('--enable_test', action='store_true', default=False, help='enable test')
  parser.add_argument('--enable_test_full', action='store_true', default=False, help='enable test full size')
  parser.add_argument('--output_directory', type=str, default=None, help='directory to store output images')
  parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to store models')

  # MPLayer
  parser.add_argument('--mpnet_n_dirs', type=int, default=None, help='number of directions')
  parser.add_argument('--mpnet_mrf_mode', type=str, default=None,
                      choices={'vanilla', 'vanilla_ft', 'TRWP', 'ISGMR', 'MeanField', 'SGM'})
  parser.add_argument('--mpnet_max_iter', type=int, default=5, help='iterations')
  parser.add_argument('--mpnet_term_weight', type=float, default=None, help='term weight')
  parser.add_argument('--mpnet_smoothness_train', type=str, default='', choices={'on', 'softmax', 'sigmoid', ''})
  parser.add_argument('--mpnet_enable_soft', action='store_true', default=False, help='enable soft bp')
  parser.add_argument('--mpnet_enable_sgm_single', action='store_true', default=False, help='enable sgm single mode')

  # Edge weights
  parser.add_argument('--mpnet_scale_list', nargs='+', type=float, default=0.5, help='scale list for edge weights')
  parser.add_argument('--mpnet_sigma', type=float, default=10, help='edge weights grad sigma')
  parser.add_argument('--use_small', action='store_true', default=False, help='use small dataset')
  parser.add_argument('--mpnet_diag_value', type=float, default=0, help='diag value of label context')
  parser.add_argument('--mpnet_smoothness_mode', type=str, default='Potts', choices={'Potts', 'TL', 'TQ', 'NPotts'})
  parser.add_argument('--context_weight_decay', type=float, default=5e-4,
                      help='w-decay for label context (default: 1e-3)')

  parser.add_argument('--enable_pairwise_net', action='store_true', default=False, help='train pairwise net by network')
  parser.add_argument('--mode', type=str, default='fully', choices={'weakly', 'fully'}, help='weakly or fully')
  parser.add_argument('--dual_loss_weight', type=float, default=0)
  parser.add_argument('--enable_adjust_val', action='store_true', default=False)
  parser.add_argument('--adjust_val_factor', type=int, default=4)
  parser.add_argument('--enable_save_unary', action='store_true', default=False)
  parser.add_argument('--edge_mode', type=str, default=None,
                      choices=['gt_edge', 'superpixel_edge', 'edge_net', 'kernel_cue_real', 'kernel_cue_binary',
                               'threshold', 'canny', 'sobel', 'edge_net_sigmoid', '', 'off'])
  parser.add_argument('--edge_pixel', type=int, default=1)
  parser.add_argument('--enable_mplayer_epoch', type=int, default=0)
  parser.add_argument('--enable_dff', action='store_true', default=False)
  parser.add_argument('--resnet_pretrained_path', type=str, default=None)
  parser.add_argument('--enable_fix_unary', action='store_true', default=False)
  parser.add_argument('--resume_unary', type=str, default=None)
  parser.add_argument('--sigmoid_scale', type=float, default=1)
  parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'Adadelta'])
  parser.add_argument('--enable_trans_init', action='store_true', default=False)
  parser.add_argument('--superpixel', type=str, default='BASS', choices=['BASS', 'CRISP', ''])
  parser.add_argument('--superpixel_threshold', type=float, default=800, choices=[0.05, 0.1, 800, 0])
  parser.add_argument('--enable_ft_single_lr', action='store_true', default=False)
  parser.add_argument('--enable_score_scale', action='store_true', default=False)
  parser.add_argument('--disable_aspp', action='store_true', default=False)
  parser.add_argument('--enable_saving_label', action='store_true', default=False)

  args = parser.parse_args()

  if (args.edge_mode is not None) and (args.edge_mode == 'off'):
    args.edge_mode = None

  mpnet_n_dirs = {'TRWP': 4, 'ISGMR': 8, 'SGM': 8, 'MeanField': 4, 'vanilla': 0}
  lr = {'TRWP': 1e-4, 'ISGMR': 1e-6, 'SGM': 1e-4, 'MeanField': 1e-6, 'vanilla': 7e-3}
  mpnet_term_weight = {'TRWP': 20, 'ISGMR': 5, 'SGM': 5, 'MeanField': 5, 'vanilla': 0}
  epochs = {'TRWP': 40, 'ISGMR': 40, 'SGM': 40, 'MeanField': 40, 'vanilla': 60}
  batch_size = {'TRWP': 12, 'ISGMR': 12, 'SGM': 12, 'MeanField': 10, 'vanilla': 12}

  if args.mpnet_n_dirs is None:
    args.mpnet_n_dirs = mpnet_n_dirs[args.mpnet_mrf_mode]

  if args.lr is None:
    args.lr = lr[args.mpnet_mrf_mode]

  if args.mpnet_term_weight is None:
    args.mpnet_term_weight = mpnet_term_weight[args.mpnet_mrf_mode]

  if args.epochs is None:
    args.epochs = epochs[args.mpnet_mrf_mode]

  if args.batch_size is None:
    args.batch_size = batch_size[args.mpnet_mrf_mode]

  if args.edge_mode != 'superpixel_edge':
    args.superpixel = ''
    args.superpixel_threshold = 0

  if not args.enable_test and args.mpnet_mrf_mode == 'vanilla':
    args.resume_unary = ''
    args.resume = ''

  if args.mpnet_mrf_mode == 'SGM':
    args.mpnet_max_iter = 1

  if args.dataset_root == '':
    args.dataset_root = None

  return args
