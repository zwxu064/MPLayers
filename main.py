import os, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from tqdm import tqdm
from dataloaders import make_data_loader
from dataloaders.custom_transforms import denormalizeimage
from modeling.sync_batchnorm.replicate import DataParallelWithCallback, patch_replication_callback
from modeling.deeplab import *
from utils import *
from mpnet import MPNet
from options import set_config, set_seed

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        args.enable_symmetric = True
        args.mpnet_smoothness_trunct_loc = -1
        args.mpnet_smoothness_trunct_value = -1
        args.enable_cuda = True
        args.n_classes = self.nclass

        model = MPNet(args)

        # Define Optimizer
        if args.optimizer in {'SGD', 'Adam'}:
            # Attention on _adjust_learning_rate in lr_scheduler.py
            if self.args.enable_fix_unary:
                train_params = [{'params': model.get_10x_lr_params(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
            else:
                if self.args.enable_ft_single_lr:
                    train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr, 'weight_decay': args.weight_decay},
                                    {'params': model.get_10x_lr_params(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
                else:
                    train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr, 'weight_decay': args.weight_decay},
                                    {'params': model.get_10x_lr_params(), 'lr': args.lr * 10, 'weight_decay': args.weight_decay}]

            if args.optimizer == 'SGD':
                optimizer = torch.optim.SGD(train_params, momentum=args.momentum, nesterov=args.nesterov)
            else:
                optimizer = torch.optim.Adam(train_params)
        elif args.optimizer == 'Adadelta':
            if self.args.enable_fix_unary:
                train_params = [{'params': model.get_10x_lr_params()}]
            else:
                train_params = [{'params': model.get_1x_lr_params()},
                                {'params': model.get_10x_lr_params()}]

            optimizer = torch.optim.Adadelta(train_params)
        else:
            assert False

        self.optimizer = optimizer

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset, server=args.server),
                                                args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset,
                                                  self.train_loader,
                                                  self.nclass,
                                                  server=args.server)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.criterion_edge = nn.MSELoss()
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        self.evaluator_single = Evaluator(self.nclass)

        # Define lr scheduler
        if (args.optimizer != 'Adadelta') and (args.lr_scheduler in {'poly', 'step', 'cos'}):
            self.scheduler = LR_Scheduler(args.lr_scheduler,
                                          args.lr,
                                          args.epochs,
                                          len(self.train_loader),
                                          enable_ft=args.ft,
                                          enable_ft_single_lr=args.enable_ft_single_lr,
                                          warmup_epochs=args.warmup_epochs)
        else:
            self.scheduler = None
            self.lr_scheduler = None

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume or args.resume_unary:
            self.run_resume(args)

        # Using cuda
        if self.args.cuda:
            if self.args.gpu_number > 1:
                self.model = DataParallelWithCallback(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        # Clear start epoch if fine-tuning
        if args.ft or args.enable_fix_unary:
            args.start_epoch = 0

    def run_resume(self, args):
        if self.args.resume_unary:
            assert os.path.exists(self.args.resume_unary)
            checkpoint = torch.load(args.resume_unary)
            state_dict = {k.replace('deeplab.', ''): v for k, v in checkpoint['state_dict'].items()}
            print("=> Loaded checkpoint '{}' (epoch {})".format(args.resume_unary, checkpoint['epoch']))
            self.model.deeplab.load_state_dict(state_dict)

        if self.args.resume:
            assert os.path.exists(self.args.resume)
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']
            self.args.start_epoch = checkpoint['epoch']
            print("=> Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            self.model.load_state_dict(state_dict)

        if (not args.ft) and (not args.enable_fix_unary):
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'optimizer' in checkpoint:
            param_groups = checkpoint['optimizer']['param_groups']
            for idx, param_group in enumerate(param_groups):
                print('=> Finetuned group {}, lr: {}'.format(idx, param_group['lr']))

        # if 'best_pred' in checkpoint:
        #     self.best_pred = checkpoint['best_pred']
        #     print('=> Resume best prediction: {}'.format(self.best_pred))

        if 'current_pred' in checkpoint:
            print('=> Resume current prediction: {}'.format(checkpoint['current_pred']))

    def training(self, epoch):
        train_loss = 0.0
        train_celoss = 0.0
        train_labelloss = 0.0
        train_edgeloss = 0.0
        duration = 0

        self.model.train()

        # For step-training, fixing unary-net
        if self.args.enable_fix_unary:
            if self.args.gpu_number == 1:
                self.model.deeplab.eval()
            else:
                self.model.module.deeplab.eval()

        # Training unary-net but fix batchnorm running_mean and running_var
        # using pretrained models (e.g., resnet-101)
        if self.args.ft:  # self.args.freeze_bn:
            if self.args.gpu_number == 1:
                self.model.freezebn_modules([self.model])
            else:
                self.model.module.freezebn_modules([self.model.module])

        num_img_tr = len(self.train_loader)

        for i in range(len(self.optimizer.param_groups)):
            print('==> LR adjusted', self.optimizer.param_groups[i]['lr'])

        # While now this is redundant given fixed unary-net
        if self.args.gpu_number == 1:
            self.model.set_enable_mplayer(epoch >= self.args.enable_mplayer_epoch)
        else:
            self.model.module.set_enable_mplayer(epoch >= self.args.enable_mplayer_epoch)

        tbar = tqdm(self.train_loader)

        for i, sample in enumerate(tbar):
            image = sample['image']
            target = sample['label'] if ('label' in sample) else None
            edge_weights = sample['edge_weights'][0] if ('edge_weights' in sample) else None
            edge = sample['edge'] if ('edge' in sample) else None

            if target is not None:
                croppings = (target != 254).float()
                target[target == 254] = 255

            if self.args.cuda:
                image = image.cuda()
                target = target.cuda() if (target is not None) else None
                edge_weights = edge_weights.cuda() if (edge_weights is not None) else None
                edge = edge.cuda() if (edge is not None) else None

            time_start = time.time()

            if self.scheduler:
                self.scheduler(self.optimizer, i, epoch, self.best_pred)

            self.optimizer.zero_grad()
            outputs = self.model(image, edge_weights=edge_weights, gt=target)
            output_unary, output_final, output_edgemap, label_context = outputs[0], outputs[1], outputs[2], outputs[3]
            model_time = time.time() - time_start
            time_start = time.time()

            # Do not Softmax/Sigmoid this in training, very low accuracy
            # Because cross_entropy already have inbuilt LogSoftmax
            if False:
                output_unary = nn.Sigmoid()(output_unary / 10)  # nn.Softmax(dim=1)(output_unary)
                output_final = nn.Sigmoid()(output_final / 10)  # nn.Softmax(dim=1)(output_final)

            celoss = self.args.dual_loss_weight * self.criterion(output_unary, target) + \
                         self.criterion(output_final, target)
            train_celoss += celoss.item()
            loss = celoss

            loss_time = time.time() - time_start
            duration += model_time + loss_time

            loss.backward()

            self.optimizer.step()
            train_loss += loss.item()

            #
            tbar.set_description('Epoch:%d, train loss:%.3f = CE:%.3f + Edge:%.3f + Label:%.3f'
                                 % (epoch,
                                    train_loss / (i + 1),
                                    train_celoss / (i + 1),
                                    train_edgeloss / (i + 1),
                                    train_labelloss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            if (i % 30 == 0) and self.args.mpnet_smoothness_train \
                and (self.args.mpnet_smoothness_train in ['sigmoid', 'softmax', 'on']):
                print('Label context viz every 30 iterations: {}'.format(label_context))

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f, time: %.4fs' % (train_loss, duration))

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, image_name = sample['image'], sample['name']
            target = sample['label'] if ('label' in sample) else None
            edge_weights = sample['edge_weights'][0] if ('edge_weights' in sample) else None
            edge = sample['edge'] if ('edge' in sample) else None

            if target is not None:
                target[target == 254] = 255

            if self.args.cuda:
                image = image.cuda()
                target = target.cuda() if (target is not None) else None
                edge_weights = edge_weights.cuda() if (edge_weights is not None) else None

            with torch.no_grad():
                edge_weight_in = None if (self.args.edge_mode in ['edge_net', 'edge_net_sigmoid']) else edge_weights
                outputs = self.model(image, edge_weights=edge_weight_in, gt=target,
                                     image_name=sample['name'], image_size=sample['size'])
                output_unary, output_final, output_edgemap, _, output_small, \
                    output_small_mp, edge_weights_merge = outputs

            loss = self.args.dual_loss_weight * self.criterion(output_unary, target) \
                       + self.criterion(output_final, target)

            test_loss += loss.item()
            tbar.set_description('Epoch:%d, valid loss:%.3f.' % (epoch, test_loss / (i + 1)))

            pred = output_final.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred_small = np.argmax(output_small.data.cpu().numpy(), axis=1)
            pred_small_mp = np.argmax(output_small_mp.data.cpu().numpy(), axis=1)
            edge = output_edgemap.squeeze(1) if (output_edgemap is not None) else edge

            self.evaluator.add_batch(target, pred)

            # Add batch sample into evaluator
            mIoU_single = []
            for idx in range(image.size(0)):
                self.evaluator_single.reset()
                self.evaluator_single.add_batch(target[idx], pred[idx])
                mIoU_single_per = self.evaluator_single.Mean_Intersection_over_Union()
                mIoU_single.append(mIoU_single_per)

            if self.args.output_directory and \
                    (os.path.exists(self.args.output_directory)) \
                    and ((not self.args.enable_save_unary and i <= 5) \
                         or self.args.enable_save_unary \
                         or (self.args.enable_test and self.args.val_batch_size == 1)):
                output_directory = os.path.join(self.args.output_directory, 'epoch{}'.format(epoch))
                unary_directory = os.path.join(output_directory, 'unary')
                if not os.path.exists(output_directory):
                    os.mkdir(output_directory)

                if self.args.enable_save_unary and not os.path.exists(unary_directory):
                    os.mkdir(unary_directory)

                n_images = pred.shape[0]

                for idx in range(n_images):
                    image_name_per = image_name[idx]
                    if edge is not None:
                        visualization(image[idx],
                                      target[idx],
                                      pred[idx],
                                      edge[idx],
                                      image_name=image_name_per,
                                      accuracy=mIoU_single[idx],
                                      save_dir=output_directory,
                                      enable_save_all=self.args.enable_test)
                    else:
                        visualization(image[idx],
                                      target[idx],
                                      pred[idx],
                                      image_name=image_name_per,
                                      accuracy=mIoU_single[idx],
                                      save_dir=output_directory,
                                      enable_save_all=self.args.enable_test)

                    if self.args.mpnet_enable_edge_weight:
                      edge_weight_per = edge_weights_merge[idx].detach()
                      visualization_png(edge_weight_per[0],
                                        edge_weight_per[2],
                                        pred_small[idx],
                                        pred_small_mp[idx],
                                        image_name=image_name_per,
                                        accuracy=mIoU_single[idx],
                                        save_dir=output_directory,
                                        enable_save_all=self.args.enable_test)

                    if self.args.enable_save_unary:
                        height_per, width_per = sample['size'][0][idx], sample['size'][1][idx]
                        unary_save = output_final[idx].permute(1, 2, 0)[:height_per, :width_per]
                        unary_save -= unary_save.min()
                        unary_save = unary_save.byte().detach().cpu().numpy()
                        unary_save_name = os.path.join(unary_directory, '{}_unary.mat'.format(image_name_per))
                        scio.savemat(unary_save_name, {'score': unary_save})

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.val_batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        # if (mIoU > self.best_pred) or (epoch % self.args.save_interval == 0):
        if mIoU > self.best_pred:
            if mIoU > self.best_pred:
                is_best = True
                self.best_pred = mIoU
            else:
                is_best = False

            state_dict_save = self.model.state_dict() if (self.args.gpu_number == 1) else self.model.module.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict_save,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'current_pred': mIoU},
                is_best, filename = 'ckpt_{}.pth.tar'.format(str(epoch + 1)))


if __name__ == "__main__":
    args = set_config()
    assert not args.enable_pairwise_net, \
        'Disable pairwise net in this branch since it needs a different version of CUDA'

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # Set seed
    set_seed(args.seed)
    args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    args.gpu_number = len(args.gpu_ids)

    # Check multiple GPUs and sync bnorm
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    if args.cuda:
      args.sync_bn = (args.gpu_number > 1)

    if args.enable_test:
        if not args.val_batch_size:
            args.val_batch_size = 1
        else:
            print('test val_batch_size: ', args.val_batch_size)
    else:
        if not args.val_batch_size:
            args.val_batch_size = args.batch_size

    if args.enable_adjust_val and (args.val_batch_size != 1):
        args.val_batch_size == 1
        print('Enable adjust val size while val batch size is {} (must be 1).'.format(args.val_batch_size))

    # Zhiwei, for unary net, set True, Important!!!
    args.freeze_bn = (args.gpu_number > 1)

    # DeepLab part 2
    args.deeplab_backbone = args.backbone
    args.deeplab_outstride = args.out_stride
    args.deeplab_sync_bn = args.sync_bn
    args.deeplab_freeze_bn = args.freeze_bn

    if args.mpnet_mrf_mode in ['TRWP', 'ISGMR', 'MeanField', 'SGM']:
        args.mpnet_scale_list = [args.mpnet_scale_list] \
            if (not isinstance(args.mpnet_scale_list, list)) else args.mpnet_scale_list
        args.mpnet_edge_weight_fn = multi_edge_weights
        args.deeplab_enable_interpolation = False
        args.mpnet_enable_edge_weight = True if args.edge_mode else False
    else:
        args.mpnet_n_dirs = 0
        args.mpnet_max_iter = 0
        args.mpnet_term_weight = 0
        args.mpnet_smoothness_train = ''
        args.mpnet_enable_soft = False
        args.mpnet_enable_edge_weight = False
        args.mpnet_scale_list = None
        args.mpnet_sigma = None
        args.mpnet_edge_weight_fn = None
        args.deeplab_enable_interpolation = True
        args.enable_mplayer_epoch = args.epochs + 2
        args.edge_mode = None
        args.enable_pairwise_net = False

    if args.use_small:
        args.enable_adjust_val = True

    if args.enable_test:
        args.batch_size = 1

    if not args.ft:
        args.enable_ft_single_lr = False

    if args.output_directory:
        args.output_directory = args.output_directory + '_withoutcrf'

        if args.enable_test:
            args.output_directory = args.output_directory + '_full'
        else:
            args.output_directory = args.output_directory + '_crop{}'.format(args.crop_size)

        if args.mpnet_mrf_mode in ['TRWP', 'ISGMR', 'MeanField', 'SGM']:
            args.output_directory = args.output_directory + '_{}'.format(args.mpnet_mrf_mode)

            if args.mpnet_enable_soft:
                args.output_directory = args.output_directory + '_soft'

        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)

    # default settings for epochs, batch_size and lr
    if not args.epochs:
        epoches = {'coco': 30, 'cityscapes': 200, 'pascal': 50}
        args.epochs = epoches[args.dataset.lower()]

    if not args.batch_size:
        args.batch_size = 4 * args.gpu_number

    if not args.lr:
        lrs = {'coco': 0.1, 'cityscapes': 0.01, 'pascal': 0.007}
        args.lr = lrs[args.dataset.lower()] / (4 * args.gpu_number) * args.batch_size

    if not args.checkname:
        args.checkname = 'deeplab-' + str(args.backbone)

    print(args)
    trainer = Trainer(args)
    print('Starting Epoch: {}, Total Epoch: {}'.format(trainer.args.start_epoch, trainer.args.epochs))

    if not args.enable_test:
        if trainer.args.start_epoch < trainer.args.epochs:
            for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
                set_seed(epoch)
                trainer.training(epoch)
                if not trainer.args.no_val:
                    trainer.validation(epoch)
        elif trainer.args.start_epoch == trainer.args.epochs:
           if not trainer.args.no_val:
               trainer.validation(trainer.args.start_epoch)
        else:
           assert False
    else:
        if args.resume and os.path.isdir(args.resume):
            model_root = args.resume

            for idx in range(args.epochs):
                args.resume = os.path.join(model_root, 'ckpt_{}.pth.tar'.format(idx + 1))
                trainer.run_resume(args)
                trainer.validation(trainer.args.start_epoch)
        else:
            trainer.validation(trainer.args.start_epoch)

    trainer.writer.close()
