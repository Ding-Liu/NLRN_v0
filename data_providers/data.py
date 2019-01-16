import tensorflow as tf
import util
import os.path


def dataset(root_folder, flist, sigma=25, residual=False, patch_size=48, epoch_num=1):
    with open(flist) as f:
        filename_list = f.read().splitlines()
    filename_list = [os.path.join(root_folder, x) for x in filename_list]
    filename_queue = tf.train.slice_input_producer([filename_list], num_epochs=epoch_num)
    image_file = tf.read_file(filename_queue[0])
    image = tf.image.decode_image(image_file, channels=1)
    # image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32)  # change data type from tf.int32 to tf.float32
    # image = tf.image.convert_image_dtype(image, tf.float32)  # change data range from 0-255 to 0-1
    patches = make_patches(image, patch_size=patch_size)

    # Add Gaussian noise
    noise = tf.random_normal(tf.shape(patches), mean=0, stddev=sigma)
    noisy_patches = patches + noise
    if residual:
        output_patches = - noise
    else:
        output_patches = patches

    return output_patches, noisy_patches


def make_patches(image, patch_size=48):
    image_list = augment_image(image)
    # patch_list = [util.image_to_patches(x) for x in image_list]
    patch_list = [tf.random_crop(x, [patch_size, patch_size, 1]) for x in image_list]
    patches = tf.stack(patch_list, axis=0)
    return patches


def augment_image(image):
    image_list = []
    image2 = tf.image.rot90(image)
    # image_list.append(tf.image.flip_up_down(image))
    # image_list.append(tf.image.flip_left_right(image))
    # image_list.append(tf.image.flip_left_right(tf.image.flip_up_down(image)))
    image_list.append(tf.image.random_flip_up_down(tf.image.random_flip_left_right(image)))
    image_list.append(tf.image.random_flip_up_down(tf.image.random_flip_left_right(image)))
    image_list.append(tf.image.random_flip_up_down(tf.image.random_flip_left_right(image2)))
    image_list.append(tf.image.random_flip_up_down(tf.image.random_flip_left_right(image2)))
    # image_list.append(image2)
    # image_list.append(tf.image.flip_up_down(image2))
    # image_list.append(tf.image.flip_left_right(image2))
    # image_list.append(tf.image.flip_left_right(tf.image.flip_up_down(image)))
    return image_list
