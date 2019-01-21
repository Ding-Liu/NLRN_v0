import tensorflow as tf
import math
import os
import scipy.misc
import scipy.io
import numpy as np
import util
import importlib
import cv2
import imresize

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_name', 'data', 'Directory to put the training data.')
flags.DEFINE_string('image_folder', '/ws/ifp-06_1/dingliu2/data/Pascal_VOC2007/VOCdevkit/VOC2007/JPEGImages/',
                    'image folder of the ground truth.')
flags.DEFINE_string('output_path', './result_sr/',
                    'output path of the super-resolved image.')
flags.DEFINE_string('model_name', 'model_conv', 'Directory of the network definition.')
flags.DEFINE_string('model_file', 'tmp/model_conv', 'Directory of the model for testing.')
flags.DEFINE_integer('scale', '2', 'scaling factor for testing')
flags.DEFINE_integer('state_num', '12', 'Number of recurrent states in model')
flags.DEFINE_integer('patch_size', '45', 'patch size for non-local operation')
flags.DEFINE_integer('batch_size', '100', 'batch size for inference')

data = importlib.import_module('data_providers.' + FLAGS.data_name)
model = importlib.import_module('models.' + FLAGS.model_name)

with tf.Graph().as_default():
    residual = True
    datafiles = [f for f in os.listdir(FLAGS.image_folder) if
                 (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPEG') or f.endswith('.bmp'))]
    datafiles.sort()
    lr_image_list = []
    hr_image_list = []
    lr_image_patch_list = []
    lr_bic_image_list = []
    stride = 7

    for f in datafiles:
        img = cv2.imread(FLAGS.image_folder + f).astype(np.float32)
        if img.ndim == 3:
            img = img[:, :, ::-1]  # BGR to RGB
            img_ycbcr = util.rgb2ycbcr(img / 255.0) * 255.0
        # img_y = img_ycbcr[:, :, 0]
        # img_y = np.rint(np.clip(img_y, 0, 255))
        # img_y = util.modcrop(img_y, FLAGS.scale)
        # img_y_l = imresize.fast_imresize(img_y, 1 / float(FLAGS.scale))
        # img_y_b = imresize.fast_imresize(img_y_l, FLAGS.scale)
        # # img_y_b = np.rint(np.clip(img_y_b, 0, 255))

            img_ycbcr = util.modcrop(img_ycbcr, FLAGS.scale)
            img_y = img_ycbcr[:, :, 0]
            img_y = np.rint(np.clip(img_y, 0, 255))
            img_ycbcr_l = imresize.fast_imresize(img_ycbcr, 1 / float(FLAGS.scale))
            img_ycbcr_b = imresize.fast_imresize(img_ycbcr_l, FLAGS.scale)
            img_y_b = img_ycbcr_b[:, :, 0]
            lr_bic_image_list.append(img_ycbcr_b)
        else:
            img_y = util.modcrop(img, FLAGS.scale)
            img_y_l = imresize.fast_imresize(img_y, 1 / float(FLAGS.scale))
            img_y_b = imresize.fast_imresize(img_y_l, FLAGS.scale)
            lr_bic_image_list.append(img_y_b)

        lr_image_list.append(img_y_b.astype(np.float32))
        hr_image_list.append(img_y.astype(np.float32))

        h_idx_list = list(range(0, hr_image_list[-1].shape[0] - FLAGS.patch_size, stride)) + [hr_image_list[-1].shape[0] - FLAGS.patch_size]
        w_idx_list = list(range(0, hr_image_list[-1].shape[1] - FLAGS.patch_size, stride)) + [hr_image_list[-1].shape[1] - FLAGS.patch_size]
        lr_patch_list = []
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                lr_patch_list.append(lr_image_list[-1][h_idx:h_idx + FLAGS.patch_size, w_idx:w_idx + FLAGS.patch_size])
        lr_image_patch_list.append(np.stack(lr_patch_list, axis=0))

    input_image = tf.placeholder(tf.float32, shape=(None, None, None))
    input_image_shape = tf.shape(input_image)
    input_image_reshaped = tf.reshape(input_image,
                                      [input_image_shape[0], input_image_shape[1], input_image_shape[2], 1])
    with tf.variable_scope("model"):
        output_image = model.build_model(model_input=input_image_reshaped, state_num=FLAGS.state_num, is_training=False)

    # init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    saver = tf.train.Saver()
    error_acc = .0
    psnr_acc = .0
    acc = 0

    with tf.Session() as sess:
        sess.run(init_local)
        if tf.gfile.Exists(FLAGS.model_file) or tf.gfile.Exists(FLAGS.model_file + '.index'):
            saver.restore(sess, FLAGS.model_file)
            print('Model restored from ' + FLAGS.model_file)
        else:
            print('Model not found')
            exit()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for i in range(len(lr_image_list)):
                batch_no = int(math.ceil(lr_image_patch_list[i].shape[0] / float(FLAGS.batch_size)))
                output_img_patch_list = []
                tmp_list = []
                for batch_id in range(batch_no):
                    cur_batch = lr_image_patch_list[i][
                                batch_id * FLAGS.batch_size:min(batch_id * FLAGS.batch_size + FLAGS.batch_size,
                                                                lr_image_patch_list[i].shape[0]), ...]
                    output_batch = sess.run(output_image, feed_dict={input_image: cur_batch})
                    output_img_patch_list.append(output_batch)

                output_img_patches = np.concatenate(output_img_patch_list, axis=0)
                h_idx_list = list(range(0, lr_image_list[i].shape[0] - FLAGS.patch_size, stride)) \
                             + [lr_image_list[i].shape[0] - FLAGS.patch_size]
                w_idx_list = list(range(0, lr_image_list[i].shape[1] - FLAGS.patch_size, stride)) \
                             + [lr_image_list[i].shape[1] - FLAGS.patch_size]

                cnt_map = np.zeros_like(lr_image_list[i])
                output_img = np.zeros_like(lr_image_list[i])
                cnt = 0
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        output_img[h_idx:h_idx + FLAGS.patch_size,
                        w_idx:w_idx + FLAGS.patch_size] += output_img_patches[cnt, :, :, :].squeeze()
                        cnt_map[h_idx:h_idx + FLAGS.patch_size, w_idx:w_idx + FLAGS.patch_size] += 1
                        cnt += 1
                output_img /= cnt_map

                if residual:
                    sr_img = output_img.squeeze() + lr_image_list[i]
                else:
                    sr_img = output_img.squeeze()

                error_per_image, psnr_per_image = util.computePSNR(util.shave(hr_image_list[i], FLAGS.scale),
                                                                   util.shave(np.rint(np.clip(sr_img, 0, 255)), FLAGS.scale))
                print(datafiles[i], error_per_image, psnr_per_image)

                if lr_bic_image_list[i].ndim == 3:
                    img_ycbcr_b = lr_bic_image_list[i]
                    img_ycbcr_b[:, :, 0] = np.clip(sr_img, 0, 255)
                    im_h = util.ycbcr2rgb(img_ycbcr_b / 255.0) * 255.0
                    scipy.misc.toimage(im_h, cmin=0, cmax=255).save(FLAGS.output_path + os.path.splitext(datafiles[i])[0] + '_sr.png')
                else:
                    # scipy.misc.imsave(FLAGS.output_path + os.path.splitext(datafiles[i])[0] + '_sr.png',
                    #           np.rint(np.clip(sr_img, 0, 255)))
                    scipy.misc.toimage(sr_img, cmin=0, cmax=255).save(FLAGS.output_path + os.path.splitext(datafiles[i])[0] + '_sr.png')

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
