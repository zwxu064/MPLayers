import matplotlib.pyplot as plt
import numpy as np
import time
from skimage import feature, color, filters
from skimage.transform import rescale
from PIL import Image
import scipy.io as scio
import glob


def run_batch(src_dir, dst_dir, year=None):
  np.random.seed(0)

  extensions = ("*.png","*.jpg","*.jpeg")
  image_files = []
  for extenstion in extensions:
    image_files.extend(glob.glob(os.path.join(src_dir, extenstion)))
  image_files = sorted(image_files)

  for image_file in image_files:
    image_name = image_file.split('/')[-1].split('.')[0]
    save_path = os.path.join(dst_dir, '{}.png'.format(image_name))

    if (args.year is None) or (save_path.find('{}_'.format(str(args.year))) > -1):
      if os.path.exists(save_path):
        continue


def run_canny(image):
  time_start = time.time()
  edge = feature.canny(image, low_threshold=None, high_threshold=None, sigma=1, use_quantiles=False, mask=None)
  print(time.time() - time_start)

  # plt.figure()
  # plt.subplot(121)
  # plt.imshow(image)
  # plt.subplot(122)
  # plt.imshow(edge)
  # plt.show()

  return edge


def run_sobel(image):
  time_start = time.time()
  edge = filters.sobel(image)
  edge = edge / edge.max()
  print(time.time() - time_start)

  # plt.figure()
  # plt.subplot(121)
  # plt.imshow(image)
  # plt.subplot(122)
  # plt.imshow(edge)
  # plt.show()

  return edge


if __name__ == '__main__':
  image_name = '2007_001423'
  image_path = "/home/xu064/Datasets/Weakly_Seg/VOC2012/JPEGImages/{}.jpg".format(image_name)
  image = np.array((Image.open(image_path).convert('L')))
  image = rescale(image, 1, anti_aliasing=True)

  canny = run_canny(image)
  sobel = run_sobel(image)

  scio.savemat('./data/{}_edges.mat'.format(image_name),
               {'canny': canny,
                'sobel': sobel})

  # Do not do this, only create edges in real time
  # run_batch('/home/xu064/Datasets/Weakly_Seg/pascal_scribble/JPEGImages',
  #           '/home/xu064/Datasets/Weakly_Seg/pascal_scribble/Edges/Canny',
  #           year=2007)
