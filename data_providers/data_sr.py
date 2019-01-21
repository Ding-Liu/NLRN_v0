import tensorflow as tf
import util
import os.path
import scipy.io
import numpy as np


def dataset(root_folder, flist, residual=False, patch_size=48, epoch_num=1):
    with open(flist) as f:
        filename_list = f.read().splitlines()
    filename_list = [os.path.join(root_folder, x) for x in filename_list]

    filename_queue = tf.train.slice_input_producer([filename_list], num_epochs=epoch_num)

    hr_patch, lr_patch = tf.py_func(read_image, [filename_queue[0], patch_size], [tf.float32, tf.float32], stateful=False)

    hr_patch = tf.reshape(hr_patch, [40, patch_size, patch_size, 1])
    lr_patch = tf.reshape(lr_patch, [40, patch_size, patch_size, 1])

    if residual:
        output_patch = hr_patch - lr_patch
    else:
        output_patch = hr_patch

    return output_patch, lr_patch


def read_image(filename, patch_size):
    try:
        data_mat = scipy.io.loadmat(filename)
    except:
        print(filename, ' not successfully loaded!')

    hr_image = data_mat['im_hr'].astype(np.float32) * 255.0
    lr_image = data_mat['im_lr'].astype(np.float32) * 255.0

    hr_patches = []
    lr_patches = []
    for _ in range(40):
        hr_patch, lr_patch = make_patches(hr_image, lr_image, patch_size=patch_size)
        hr_patches.append(hr_patch)
        lr_patches.append(lr_patch)
    hr_patches = np.stack(hr_patches)
    lr_patches = np.stack(lr_patches)
    return hr_patches, lr_patches


def make_patches(hr_image, lr_image, patch_size=48):

    # h_idx = tf.random_uniform([1], 0, tf.shape(hr_image)[0]-patch_size, dtype=tf.int32)
    # w_idx = tf.random_uniform([1], 0, tf.shape(hr_image)[1]-patch_size, dtype=tf.int32)
    h_idx = np.random.randint(0, hr_image.shape[0]-patch_size)
    w_idx = np.random.randint(0, hr_image.shape[1]-patch_size)

    hr_patch = hr_image[h_idx:h_idx+patch_size, w_idx:w_idx+patch_size]
    lr_patch = lr_image[h_idx:h_idx+patch_size, w_idx:w_idx+patch_size]

    seed = np.random.randint(0, 7)
    hr_patch = apply_transform(hr_patch, seed)
    lr_patch = apply_transform(lr_patch, seed)

    return hr_patch, lr_patch


def apply_transform(image, seed):
    if seed == 0:
        res = image
    elif seed == 1:
        res = np.flipud(image)
    elif seed == 2:
        res = np.fliplr(image)
    elif seed == 3:
        res = np.fliplr(np.flipud(image))
    elif seed == 4:
        res = np.rot90(image)
    elif seed == 5:
        res = np.flipud(np.rot90(image))
    elif seed == 6:
        res = np.fliplr(np.rot90(image))
    elif seed == 7:
        res = np.fliplr(np.flipud(np.rot90(image)))

    return res


if __name__ == '__main__':
    dataset('/ws/ifp-06_1/dingliu2/data/SR/BSDS500/data/images/',
            '/ws/ifp-06_1/dingliu2/data/SR/BSDS500/data/images/train_test.list')
