# ---------------------------------------------------------
# Python Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import random
import numpy as np
import matplotlib as mpl
import scipy.misc
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


class ImagePool(object):
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.imgs = []

    def query(self, img):
        if self.pool_size == 0:
            return img

        if len(self.imgs) < self.pool_size:
            self.imgs.append(img)
            return img
        else:
            if random.random() > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp_img = self.imgs[random_id].copy()
                self.imgs[random_id] = img.copy()
                return tmp_img
            else:
                return img


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def imagefiles2arrs(filenames):
    img_shape = image_shape(filenames[0])
    images_arr = None

    if len(img_shape) == 3:  # color image
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape) == 2:  # gray scale image
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)

    for file_index in range(len(filenames)):
        img = Image.open(filenames[file_index])
        images_arr[file_index] = np.asarray(img).astype(np.float32)

    return images_arr


def image_shape(filename):
    img = Image.open(filename, mode="r")
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape


def print_metrics(itr, kargs):
    print("*** Iteration {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value))
    print("")
    sys.stdout.flush()


def transform(img):
    return img / 127.5 - 1.0


def inverse_transform(img):
    return (img + 1.) / 2.


def preprocess_pair(img_a, img_b, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_a = scipy.misc.imresize(img_a, [fine_size, fine_size])
        img_b = scipy.misc.imresize(img_b, [fine_size, fine_size])
    else:
        img_a = scipy.misc.imresize(img_a, [load_size, load_size])
        img_b = scipy.misc.imresize(img_b, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        img_a = img_a[h1:h1 + fine_size, w1:w1 + fine_size]
        img_b = img_b[h1:h1 + fine_size, w1:w1 + fine_size]

        if flip and np.random.random() > 0.5:
            img_a = np.fliplr(img_a)
            img_b = np.fliplr(img_b)

    return img_a, img_b


def imread(path, is_gray_scale=False, img_size=None):
    if is_gray_scale:
        img = scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        img = scipy.misc.imread(path, mode='RGB').astype(np.float)

        if not (img.ndim == 3 and img.shape[2] == 3):
            img = np.dstack((img, img, img))

    if img_size is not None:
        img = scipy.misc.imresize(img, img_size)

    return img


def load_image(image_path, which_direction=0, is_gray_scale=True, img_size=(256, 256, 1)):
    input_img = imread(image_path, is_gray_scale=is_gray_scale, img_size=img_size)
    w_pair = int(input_img.shape[1])
    w_single = int(w_pair / 2)

    if which_direction == 0:    # A to B
        img_a = input_img[:, 0:w_single]
        img_b = input_img[:, w_single:w_pair]
    else:                       # B to A
        img_a = input_img[:, w_single:w_pair]
        img_b = input_img[:, 0:w_single]

    return img_a, img_b


def load_data(image_path, is_gray_scale=False):
    img = imread(path=image_path, is_gray_scale=is_gray_scale)
    img_trans = transform(img)  # from [0, 255] to [-1., 1.]

    if is_gray_scale and (img_trans.ndim == 2):
        img_trans = np.expand_dims(img_trans, axis=2)

    return img_trans


def plots(imgs, iter_time, save_file, grid_cols, grid_rows, sample_batch, name=None):
    # parameters for plot size
    scale, margin = 0.02, 0.02

    # save more bigger image
    img_h, img_w, img_c = imgs.shape[1:]
    fig = plt.figure(figsize=(img_w * grid_cols * scale, img_h * grid_rows * scale))  # (column, row)
    gs = gridspec.GridSpec(grid_rows, grid_cols)  # (row, column)
    gs.update(wspace=margin, hspace=margin)

    for img_idx in range(sample_batch):
        ax = plt.subplot(gs[img_idx])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        if imgs[img_idx].shape[2] == 1:  # gray scale
            plt.imshow((imgs[img_idx]).reshape(img_h, img_w), cmap='Greys_r')
        else:
            plt.imshow((imgs[img_idx]).reshape(img_h, img_w, img_c), cmap='Greys_r')

    plt.savefig(save_file + '/{}_{}.png'.format(str(iter_time), name), bbox_inches='tight')
    plt.close(fig)


def _merge(images, size, resize_ratio=1.):
    h, w = images.shape[1], images.shape[2]
    h_ = int(h * resize_ratio)
    w_ = int(w * resize_ratio)

    img_canvas = np.zeros((h_ * size[0], w_ * size[1]))
    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])

        image_resize = scipy.misc.imresize(image, size=(h_, w_), interp='bicubic')
        img_canvas[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_resize

    return img_canvas


# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id_, iter_time, save_file, z_range=2):
    num_labels = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id_, 1), marker='o', edgecolor='none',
                cmap=discrete_cmap(num_labels, 'jet'))
    plt.colorbar(ticks=range(num_labels))
    axes = plt.gca()
    axes.set_xlim([-z_range-2, z_range+2])
    axes.set_ylim([-z_range-2, z_range+2])
    plt.grid(True)
    plt.savefig(os.path.join(save_file, 'embedding_{}.png'.format(str(iter_time))))


# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
