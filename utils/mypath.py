class Path(object):
    @staticmethod
    def db_root_dir(dataset, server='039614'):
        if dataset == 'pascal':
            # folder that contains pascal/. It should have three subdirectories 
            # called "JPEGImages", "SegmentationClassAug", and "pascal_2012_scribble" 
            # containing RGB images, groundtruth, and scribbles respectively.
            if server == '039614':
                return '/mnt/scratch/zhiwei/Code/rloss/data/pascal_scribble/'
            else:
                return '/home/xu064/WorkSpace/git-lab/pytorch-projects/MPLayers/datasets/pascal_scribble/'
        elif dataset == 'sbd':
            if server == '039614':
                return '/home/users/u5710355/Datasets/PASCAL/benchmark_RELEASE/'  # folder that contains dataset/.
            else:
                return '/home/xu064/WorkSpace/git-lab/pytorch-projects/MPLayers/datasets/PASCAL/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'dff_sbd':
            return '/home/xu064/Results/DFF/data_proc'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
