import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.signal


def resize_func_scipy(image, target_shape):
    def resize_batch(image, target_shape):
        resized = []
        for i in range(image.shape[0]):
            resized.append(scipy.misc.imresize(image[i], target_shape, interp='bicubic'))
        return np.stack(resized)
    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
    image = tf.py_func(resize_batch, [image, target_shape], tf.uint8, stateful=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

resize_func = resize_func_scipy


def image_to_patches(image, patch_height=48, patch_width=48, patch_overlap=12):
    # patch_height = 48 / scale
    # patch_width = 48 / scale
    # patch_overlap = 12 / scale
    patches = tf.extract_image_patches(image, [1, patch_height, patch_width, 1], [1, patch_height - 2 * patch_overlap, patch_width - 2 * patch_overlap, 1], [1, 1, 1, 1], padding='VALID')
    return tf.reshape(patches, [tf.shape(patches)[0] * tf.shape(patches)[1] * tf.shape(patches)[2], patch_height, patch_width, 1])


def crop_center(image, target_shape):
    origin_shape = tf.shape(image)[1:3]
    return tf.slice(image, [0, (origin_shape[0] - target_shape[0]) / 2, (origin_shape[1] - target_shape[1]) / 2, 0], [-1, target_shape[0], target_shape[1], -1])


def crop_by_pixel(x, num):
    shape = tf.shape(x)[1:3]
    return tf.slice(x, [0, num, num, 0], [-1, shape[0] - 2 * num, shape[1] - 2 * num, -1])


def pad_boundary(image, boundary_size=15):
    return tf.pad(image, [[0, 0], [boundary_size, boundary_size], [boundary_size, boundary_size], [0, 0]], mode="SYMMETRIC")


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def computePSNR(im1, im2):
    """
    im1: float np array in [0, 255]:
    im2: float np array in [0, 255]:
    """
    im1_uint8 = np.rint(np.clip(im1, 0, 255))
    im2_uint8 = np.rint(np.clip(im2, 0, 255))

    # im1_uint8 = np.clip(im1, 0, 255)
    # im2_uint8 = np.clip(im2, 0, 255)

    diff = np.abs(im1_uint8 - im2_uint8).flatten()
    rmse = np.sqrt(np.mean(np.square(diff)))
    psnr = 20 * np.log10(255.0 / rmse)
    return rmse, psnr


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def ssim(img1, img2):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    img1 = np.rint(np.clip(img1, 0, 255))
    img2 = np.rint(np.clip(img2, 0, 255))

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    mu1 = scipy.signal.fftconvolve(img1, window, mode='valid')
    mu2 = scipy.signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = scipy.signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = scipy.signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = scipy.signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)


def optimistic_restore(sess, ckpt_file):
    reader = tf.train.NewCheckpointReader(ckpt_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0])
                        for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                            tf.global_variables()),
                        tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            # if var_shape == saved_shapes[saved_var_name]:
            if var_shape == saved_shapes[saved_var_name] and curr_var.name.startswith('model/'):
                restore_vars.append(curr_var)
                print('- restoring variable: {}'
                            .format(curr_var.name))
    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, ckpt_file)


def shave(image, border):
    assert border > 0
    image = image[border:-border, border:-border, ...]
    return image
