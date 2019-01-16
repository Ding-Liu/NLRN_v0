import tensorflow as tf
import math
# import util
import os
import scipy.misc
import scipy.io
import numpy as np
import util
import importlib

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_name', 'data', 'Directory to put the training data.')
flags.DEFINE_string('image_folder', '/ws/ifp-06_1/dingliu2/data/Pascal_VOC2007/VOCdevkit/VOC2007/JPEGImages/',
                    'image folder of the ground truth.')
flags.DEFINE_string('noisy_image_folder', '/ws/ifp-06_1/dingliu2/data/Pascal_VOC2007/VOCdevkit/VOC2007/JPEGImages/',
                    'folder of the MAT files of noisy images.')
flags.DEFINE_string('output_path', './result/',
                    'output path of the denoised image.')
flags.DEFINE_string('model_name', 'model_conv', 'Directory of the network definition.')
flags.DEFINE_string('model_file', 'tmp/model_conv', 'Directory of the model for testing.')
flags.DEFINE_integer('patch_size', '45', 'patch size for non-local operation')
flags.DEFINE_integer('batch_size', '100', 'batch size for inference')
flags.DEFINE_integer('state_num', '12', 'Number of recurrent states in model')

#data = __import__(FLAGS.data_name)
#model = __import__(FLAGS.model_name)
data = importlib.import_module('data_providers.' + FLAGS.data_name)
model = importlib.import_module('models.' + FLAGS.model_name)

with tf.Graph().as_default():
    residual = True
    datafiles = [f for f in os.listdir(FLAGS.image_folder) if
                 (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPEG') or f.endswith('.bmp'))]
    datafiles.sort()
    image_list = []
    noisy_image_list = []
    noisy_image_patch_list = []
    # stride = FLAGS.patch_size // 2
    stride = 7

    for f in datafiles:
        img = scipy.misc.imread(FLAGS.image_folder + f).astype(np.float32)
        image_list.append(img)
        file_name = os.path.basename(FLAGS.image_folder + f)
        file_basename, file_extension = os.path.splitext(file_name)

        dtmp = scipy.io.loadmat(FLAGS.noisy_image_folder + file_basename + '.mat')
        noisy_img = dtmp['image'].astype(np.float32) * 255.0  # range to be 0~255
        noisy_image_list.append(noisy_img)
        h_idx_list = list(range(0, noisy_img.shape[0] - FLAGS.patch_size, stride)) + [noisy_img.shape[0] - FLAGS.patch_size]
        w_idx_list = list(range(0, noisy_img.shape[1] - FLAGS.patch_size, stride)) + [noisy_img.shape[1] - FLAGS.patch_size]
        patch_list = []
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                patch_list.append(noisy_img[h_idx:h_idx + FLAGS.patch_size, w_idx:w_idx + FLAGS.patch_size])
        noisy_image_patch_list.append(np.stack(patch_list, axis=0))

    # input_image = tf.placeholder(tf.float32, shape=(None, None))
    input_image = tf.placeholder(tf.float32, shape=(None, None, None))
    input_image_shape = tf.shape(input_image)
    # input_image_reshaped = tf.reshape(input_image, [1, input_image_shape[0], input_image_shape[1], 1])
    input_image_reshaped = tf.reshape(input_image,
                                      [input_image_shape[0], input_image_shape[1], input_image_shape[2], 1])
    with tf.variable_scope("model"):
        output_image = model.build_model(model_input=input_image_reshaped, state_num=FLAGS.state_num, is_training=False)
        # output_image, o1, o2, o3, o4 = model.build_model(input_image_reshaped, training=False)
        # output_image_1, output_image_2 = model.build_model(input_image_reshaped, training=False)
        # output_image_1, output_image_2, output_image_3 = model.build_model(input_image_reshaped, training=False)

    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    # model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
    # saver = tf.train.Saver(var_list=model_vars)
    saver = tf.train.Saver()
    error_acc = .0
    psnr_acc = .0
    acc = 0

    mask1_list = []
    mask2_list = []

    with tf.Session() as sess:
        sess.run(init_local)
        if tf.gfile.Exists(FLAGS.model_file) or tf.gfile.Exists(FLAGS.model_file + '.index'):
            saver.restore(sess, FLAGS.model_file)
            print ('Model restored from ' + FLAGS.model_file)
        else:
            print ('Model not found')
            exit()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for i in range(len(noisy_image_list)):
                # error_per_image = sess.run(error, feed_dict={input_image: noisy_image_list[i],
                #                                              noiseless_image: image_list[i]})
                # psnr_per_image = 20 * np.log10(255.0 / np.sqrt(error_per_image))

                # output_img = sess.run(output_image, feed_dict={input_image: noisy_image_list[i]})

                batch_no = int(math.ceil(noisy_image_patch_list[i].shape[0] / float(FLAGS.batch_size)))
                output_img_patch_list = []
                tmp_list = []
                for batch_id in range(batch_no):
                    # print min(batch_id*100+100, noisy_image_patch_list[i].shape[0])
                    cur_batch = noisy_image_patch_list[i][
                                batch_id * FLAGS.batch_size:min(batch_id * FLAGS.batch_size + FLAGS.batch_size,
                                                                noisy_image_patch_list[i].shape[0]), ...]
                    output_batch = sess.run(output_image, feed_dict={input_image: cur_batch})
                    # output_batch, tmp1, tmp2, tmp3, tmp4 = sess.run([output_image, o1, o2, o3, o4], feed_dict={input_image: cur_batch})
                    output_img_patch_list.append(output_batch)

                output_img_patches = np.concatenate(output_img_patch_list, axis=0)
                h_idx_list = list(range(0, noisy_image_list[i].shape[0] - FLAGS.patch_size, stride)) \
                             + [noisy_image_list[i].shape[0] - FLAGS.patch_size]
                w_idx_list = list(range(0, noisy_image_list[i].shape[1] - FLAGS.patch_size, stride)) \
                             + [noisy_image_list[i].shape[1] - FLAGS.patch_size]

                cnt_map = np.zeros_like(noisy_image_list[i])
                output_img = np.zeros_like(noisy_image_list[i])
                cnt = 0
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        output_img[h_idx:h_idx + FLAGS.patch_size,
                        w_idx:w_idx + FLAGS.patch_size] += output_img_patches[cnt, :, :, :].squeeze()
                        cnt_map[h_idx:h_idx + FLAGS.patch_size, w_idx:w_idx + FLAGS.patch_size] += 1
                        cnt += 1
                output_img /= cnt_map

                if residual:
                    denoised_img = output_img.squeeze() + noisy_image_list[i]
                else:
                    denoised_img = output_img.squeeze()
                error_per_image, psnr_per_image = util.computePSNR(image_list[i], denoised_img)
                print (datafiles[i], error_per_image, psnr_per_image)

                # scipy.misc.imsave(FLAGS.output_path + os.path.splitext(datafiles[i])[0] + '_denoised.png',
                #           np.rint(np.clip(denoised_img, 0, 255)))
                scipy.misc.toimage(denoised_img, cmin=0, cmax=255).save(FLAGS.output_path + os.path.splitext(datafiles[i])[0] + '_denoised.png')                

                error_acc += error_per_image
                psnr_acc += psnr_per_image
                acc += 1
        except tf.errors.OutOfRangeError:
            print('Done validation -- epoch limit reached')
        finally:
            coord.request_stop()
        print ('-----')
        print ('Average MSE: %.4f, Average PSNR: %.4f' % (error_acc / acc, psnr_acc / acc))
        print ('-----')
