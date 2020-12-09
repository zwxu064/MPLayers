import shutil, glob, os, sys, copy, torch, argparse
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension


parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode', type=str, choices={'stereo', 'segmentation'}, default='stereo')
args, _ = parser.parse_known_args()

argv_list = copy.deepcopy(sys.argv)
for v in argv_list:
  if v.find('--mode') > -1:
    sys.argv.pop(sys.argv.index(v))
  elif v.find('--install-dir') > -1:
    dir_name = 'Stereo' if (args.mode == 'stereo') else 'Segmentation'
    arg_dir_path = os.path.join(v, dir_name)
    dir_path = arg_dir_path.split('=')[-1]
    os.makedirs(dir_path) if not os.path.exists(dir_path) else None
    sys.argv[sys.argv.index(v)] = arg_dir_path

enable_cuda = torch.cuda.is_available()
torch_version_major = int(torch.__version__.split('.')[0])  # major.min.patch
MAX_DISPARITY = int(96) if (args.mode == 'stereo') else int(32)

include_dir = ['aux', '../tools/cpp',
               '/mnt/scratch/zhiwei/Installations/anaconda3/envs/train-cuda/include',
               '/mnt/scratch/zhiwei/Installations/anaconda3/envs/train-cuda/include/opencv',
               '/local/Installation/anaconda3/envs/pytorch3.6-env/include',
               '/local/Installation/anaconda3/envs/pytorch3.6-env/include/opencv',
               '/apps/opencv/3.4.3/include',
               '/apps/opencv/3.4.3/include/opencv']
library_dir = ['/mnt/scratch/zhiwei/Installations/anaconda3/envs/train-cuda/lib',
               '/local/Installation/anaconda3/envs/pytorch3.6-env/lib',
               '/apps/opencv/3.4.3/lib64']  # adding opencv library data61 s useless, adding in bashrc

setup(name='compute_terms',
      ext_modules=[CppExtension(name='compute_terms',
                                sources=['aux/auxPy.cpp',
                                         'aux/aux.cpp',
                                         '../tools/cpp/utils.cpp'],
                                extra_compile_args={'cxx':['-std=c++11',
                                                           '-Wno-deprecated-declarations',
                                                           '-O3']},
                                define_macros=[('USE_OPENCV', None),
                                               ('USE_FLOAT', None),
                                               ('TORCH_VERSION_MAJOR', torch_version_major)],
                                include_dirs=include_dir,
                                library_dirs=library_dir,
                                libraries=['opencv_highgui',
                                           'opencv_core',
                                           'opencv_imgproc',
                                           'opencv_imgcodecs'])],
      cmdclass={'build_ext': BuildExtension})

if not enable_cuda:
  setup(name='ISGMR',
        ext_modules=[CppExtension(name='ISGMR',
                                  sources=['ISGMR/ISGMR.cpp',
                                           'common.cpp',
                                           '../tools/cpp/utils.cpp'],
                                  extra_compile_args={'nvcc': ['-O3',
                                                               '-arch=sm_35',
                                                               '--expt-relaxed-constexpr'],
                                                      'cxx': ['-g',
                                                              '-std=c++11',
                                                              '-Wno-deprecated-declarations',
                                                              '-fopenmp']},
                                  define_macros=[('TORCH_VERSION_MAJOR', torch_version_major),
                                                 ('MAX_DISPARITY', MAX_DISPARITY)],
                                  undef_macros=['USE_CUDA'])],
        cmdclass={'build_ext': BuildExtension},
        include_dirs=['.', '../tools/cpp'])

  setup(name='TRWP',
        ext_modules=[CppExtension(name='TRWP',
                                  sources=['TRWP/TRWP.cpp',
                                           'common.cpp',
                                           '../tools/cpp/utils.cpp'],
                                  extra_compile_args={'nvcc': ['-O3',
                                                               '-arch=sm_35',
                                                               '--expt-relaxed-constexpr'],
                                                      'cxx': ['-g',
                                                              '-std=c++11',
                                                              '-Wno-deprecated-declarations',
                                                              '-fopenmp']},
                                  define_macros=[('TORCH_VERSION_MAJOR', torch_version_major),
                                                 ('MAX_DISPARITY', MAX_DISPARITY)],
                                  undef_macros=['USE_CUDA'])],
        cmdclass={'build_ext': BuildExtension},
        include_dirs=['.', '../tools/cpp'])
else:
  setup(name='ISGMR',
        ext_modules=[CUDAExtension(name='ISGMR',
                                   sources=['ISGMR/ISGMR.cpp',
                                            'common.cpp',
                                            'ISGMR/ISGMR_forward.cu',
                                            'ISGMR/ISGMR_backward.cu',
                                            '../tools/cpp/utils.cpp',
                                            '../tools/cuda/cudaUtils.cu'],
                                   extra_compile_args={'nvcc': ['-O3',
                                                                '-arch=sm_35',
                                                                '--expt-relaxed-constexpr'],
                                                       'cxx': ['-g',
                                                               '-std=c++11',
                                                               '-Wno-deprecated-declarations',
                                                               '-fopenmp']},
                                   define_macros=[('USE_CUDA', None),
                                                  ('TORCH_VERSION_MAJOR', torch_version_major),
                                                  ('MAX_DISPARITY', MAX_DISPARITY)],
                                   undef_macros=[])],
        cmdclass={'build_ext': BuildExtension},
        include_dirs=['.', '../tools/cpp', '../tools/cuda'])

  setup(name='TRWP',
        ext_modules=[CUDAExtension(name='TRWP',
                                   sources=['TRWP/TRWP.cpp',
                                            'common.cpp',
                                            'TRWP/TRWP_forward.cu',
                                            'TRWP/TRWP_backward.cu',
                                            '../tools/cpp/utils.cpp',
                                            '../tools/cuda/cudaUtils.cu'],
                                   extra_compile_args={'nvcc': ['-O3',
                                                                '-arch=sm_35',
                                                                '--expt-relaxed-constexpr'],
                                                       'cxx': ['-g',
                                                               '-std=c++11',
                                                               '-Wno-deprecated-declarations',
                                                               '-fopenmp']},
                                   define_macros=[('USE_CUDA', None),
                                                  ('TORCH_VERSION_MAJOR', torch_version_major),
                                                  ('MAX_DISPARITY', MAX_DISPARITY)],
                                   undef_macros=[])],
        cmdclass={'build_ext': BuildExtension},
        include_dirs=['.', '../tools/cpp', '../tools/cuda'])

target_dir = 'lib_stereo_slim' if (args.mode == 'stereo') else 'lib_seg_slim'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

for file in glob.glob('*.so') + glob.glob('*.egg-info'):
  file_path = os.path.join(target_dir, file)

  if os.path.exists(file_path):
    if os.path.isfile(file_path):
      os.remove(file_path)
    elif os.path.isdir(file_path):
      shutil.rmtree(file_path)

  shutil.move(file, target_dir)
