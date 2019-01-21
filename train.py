import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import util
import importlib
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_name', 'data_residual',
                    'Directory to put the training data.')
flags.DEFINE_string('root_folder', '/ws/ifp-06_1/dingliu2/data/Pascal_VOC2007/VOCdevkit/VOC2007/JPEGImages/',
                    'root folder of the training data.')
flags.DEFINE_string('flist', '/ws/ifp-06_1/dingliu2/DeepDenoising-v2/data/Pascal_VOC2007_images.txt',
                    'file list put the training data.')
flags.DEFINE_integer('sigma', '25', 'standard deviation of Gaussian noise for training')
flags.DEFINE_string('model_name', 'model_resnet_up',
                    'Directory to put the training data.')
flags.DEFINE_string('model_file_in', 'tmp/model_conv',
                    'Directory to put the training data.')
flags.DEFINE_string('model_file_out', 'tmp/model_conv',
                    'Directory to put the training data.')
flags.DEFINE_float('learning_rate', '0.001', 'Learning rate for training')
flags.DEFINE_integer('batch_size', '16', 'batch size for training')
flags.DEFINE_integer('patch_size', '45', 'patch size for training')
flags.DEFINE_boolean('mem_growth', True, 'If true, use gpu memory on demand.')
flags.DEFINE_integer('smoothed_loss_batch_num', '1000', 'mini-batch number of smoothed loss')
flags.DEFINE_integer('snapshot_batch_num', '25000', 'mini-batch number for snapshot')
flags.DEFINE_boolean('single_pixel', False, 'If true, only predict the center single pixel.')
flags.DEFINE_integer('state_num', '12', 'Number of recurrent states in model')
flags.DEFINE_boolean('continue_training', True, 'If true, continue training from a checkpoint')
flags.DEFINE_string('log_dir', './tfboard_logs', 'TensorBoard logs are stored here')

data = importlib.import_module('data_providers.' + FLAGS.data_name)
model = importlib.import_module('models.' + FLAGS.model_name)

g = tf.Graph()

with g.as_default():
    with tf.device('/cpu:0'):
        target_patches, source_patches = data.dataset(
            FLAGS.root_folder,
            FLAGS.flist,
            FLAGS.sigma,
            residual=True,
            patch_size=FLAGS.patch_size,  # <=96 for Pascal VOC2007!
            epoch_num=1e5  # 80
            )
        target_batch_staging, source_batch_staging = tf.train.shuffle_batch(
            [target_patches, source_patches],
            FLAGS.batch_size,
            32768*2,
            8192*2,
            num_threads=4,
            enqueue_many=True)
    stager = data_flow_ops.StagingArea(
        [tf.float32, tf.float32],
        shapes=[[None, None, None, 1], [None, None, None, 1]])
    stage = stager.put([target_batch_staging, source_batch_staging])
    target_batch, source_batch = stager.get()
    with tf.variable_scope("model"):
        predict_batch = model.build_model(
            model_input=source_batch, state_num=FLAGS.state_num, is_training=True)

    # learning rate schedule
    global_step = tf.Variable(0, trainable=False)
    step_num_batch = 75000  # 100000
    boundaries = [_ + step_num_batch for _ in range(0, step_num_batch*6, step_num_batch)]
    # boundaries = range(0, step_num_batch*6, step_num_batch)
    values = [FLAGS.learning_rate * (0.5**i) for i in range(6+1)]

    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    with tf.variable_scope("solver"):
        if FLAGS.single_pixel:
            loss = tf.losses.mean_squared_error(target_batch[:, tf.shape(target_batch)[1]//2-2:tf.shape(target_batch)[1]//2+3, tf.shape(target_batch)[1]//2-2:tf.shape(target_batch)[2]//2+3, :],
                                                predict_batch[:, tf.shape(predict_batch)[1]//2-2:tf.shape(predict_batch)[1]//2+3, tf.shape(predict_batch)[1]//2-2:tf.shape(predict_batch)[2]//2+3, :])  # L2 loss
        else:
            loss = tf.losses.mean_squared_error(target_batch, predict_batch)  # L2 loss
        # loss = tf.losses.absolute_difference(target_batch, predict_batch)  # L1 loss
        # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gvs = optimizer.compute_gradients(loss)
            # gradient clipping
            capped_gvs = [(tf.clip_by_norm(grad, 2.5), var) for grad, var in gvs]
            optimizer = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    tf.summary.FileWriter(FLAGS.log_dir, g).close()

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    saver2 = tf.train.Saver(max_to_keep=50)

    avg_loss_acc = .0
    cnt = 0  # counter for mini-batch
    smoothed_loss_batch_num = FLAGS.smoothed_loss_batch_num
    snapshot_batch_num = FLAGS.snapshot_batch_num
    loss_list = [None] * smoothed_loss_batch_num  # running average of loss of the latest 1000 mini-batches

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = FLAGS.mem_growth
    with tf.Session(config=config) as sess:
        sess.run(init_local)
        if (tf.gfile.Exists(FLAGS.model_file_out) or
                tf.gfile.Exists(FLAGS.model_file_out + '.index')):
            print('Model exists! Ending...')
            quit()
        if (tf.gfile.Exists(FLAGS.model_file_in) or
                tf.gfile.Exists(FLAGS.model_file_in + '.index')):
            if FLAGS.continue_training:
                saver2.restore(sess, FLAGS.model_file_in)
                print('Continue training. Model restored from ' + FLAGS.model_file_in)
            else:  # only load part of weights from the old model
                sess.run(init_global)
                util.optimistic_restore(sess, FLAGS.model_file_in)
                print('Weights loaded from ' + FLAGS.model_file_in)
        else:
            sess.run(init_global)
            print('Model initialized')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            sess.run(stage)
            start_time = time.time()
            while not coord.should_stop():
                # _, _, training_loss = sess.run(
                #     [stage, optimizer, loss])
                _, _, training_loss, lr, g_step = sess.run(
                    [stage, optimizer, loss, learning_rate, global_step])

                # print training_loss, cnt
                avg_loss_acc += training_loss

                idx = cnt % smoothed_loss_batch_num
                loss_list[idx] = training_loss
                if (cnt+1) % smoothed_loss_batch_num == 0:
                    end_time = time.time()
                    smoothed_loss = sum(loss_list) / float(smoothed_loss_batch_num)
                    batch_time = smoothed_loss_batch_num / float(end_time - start_time)
                    print ('%s batch num: %d, lr: %g, smoothed loss: %7.3f, %.4f batch/sec' %
                           (time.ctime(), cnt+1, lr, smoothed_loss, batch_time))
                    start_time = time.time()

                if cnt % snapshot_batch_num == 0 and cnt > 0:
                    print ('saving models %s at iteration %d' % (FLAGS.model_file_out + '-' + str(cnt), cnt))
                    saver2.save(sess, FLAGS.model_file_out + '-' + str(cnt))

                cnt += 1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        print('Average loss: ' + str(avg_loss_acc / cnt))
        saver2.save(sess, FLAGS.model_file_out)
        print('Model saved to ' + FLAGS.model_file_out)
