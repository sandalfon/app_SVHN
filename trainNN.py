# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:56:15 2018

@author: Sandalfon
"""

import tensorflow as tf
import os
from datetime import datetime
import time
from ModelCompletLayer import Model
from dataRefine import DataRefine
from evalu import Evalu
import utils

tf.app.flags.DEFINE_string('data_dir', '..\\data', 'Directory to read pickle file')
tf.app.flags.DEFINE_string('train_logdir', '..\\logs\\train', 'Directory to write training session')
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           'Path to restore training session (without postfix), e.g. ./logs/train/model.ckpt-100')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Default 32')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Default 1e-2')
tf.app.flags.DEFINE_integer('patience', 100, 'Default 100, set -1 to train infinitely')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'Default 10000')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'Default 0.9')
tf.app.flags.DEFINE_integer('conv_kern', 5, 'convolution kernels') 
tf.app.flags.DEFINE_integer('fil_1', 48, 'Unit spatial location 1')
tf.app.flags.DEFINE_integer('fil_2', 64, 'Unit spatial location 2')
tf.app.flags.DEFINE_integer('fil_3', 128, 'Unit spatial location 3')
tf.app.flags.DEFINE_integer('fil_4', 160, 'Unit spatial location 4')
tf.app.flags.DEFINE_integer('fil_5', 192, 'Unit spatial location 5')
tf.app.flags.DEFINE_integer('fil_fc',  3072, 'Unit spatial location layer > 5')
tf.app.flags.DEFINE_integer('fl_l', 7, 'Unit spatial location length digit sequence')
tf.app.flags.DEFINE_integer('fl_dig', 11, 'Unit spatial location digit')
tf.app.flags.DEFINE_integer('stride_odd', 2, 'Stride odd layer')
tf.app.flags.DEFINE_integer('stride_even', 1, 'Stride even layer')
tf.app.flags.DEFINE_integer('mp', 2, 'Maxpooling size')
tf.app.flags.DEFINE_float('drop_rate',  0.8, 'Drop rate')
tf.app.flags.DEFINE_float('mu',  0, 'mean')
tf.app.flags.DEFINE_float('sigma', 0.1, 'std deviation')
                  
FLAGS = tf.app.flags.FLAGS

def _run(path_to_pickle_file, path_to_train_log_dir, 
           path_to_restore_checkpoint_file,
           training_options, model_options):
    batch_size = training_options['batch_size']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000
    
    data=DataRefine()
    data.load_pickles(path_to_pickle_file)
    with tf.Graph().as_default():
        image_batch, length_batch, digits_batch = utils.build_batch(
                  data.train_dataset, data.train_labels,
                 batch_size=batch_size, shuffled=True)
                  
          
        length_logtis, digits_logits = Model.inference(image_batch, model_options)
        loss = Model.loss(length_logtis, digits_logits, length_batch, digits_batch)
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(training_options['learning_rate'], global_step=global_step,
                                                   decay_steps=training_options['decay_steps'], 
                                                   decay_rate=training_options['decay_rate'], staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.image('image', image_batch)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()  
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(path_to_train_log_dir, sess.graph)
            evalu = Evalu(os.path.join(path_to_train_log_dir, 'eval/val'))

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver = tf.train.Saver()
            if path_to_restore_checkpoint_file is not None:
                assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), \
                    '%s not found' % path_to_restore_checkpoint_file
                saver.restore(sess, path_to_restore_checkpoint_file)
                print('Model restored from file: %s' % path_to_restore_checkpoint_file)

            print('Start training')
            patience = initial_patience
            best_accuracy = 0.0
            duration = 0.0

            while True:
                start_time = time.time()
                _, loss_val, summary_val, global_step_val, learning_rate_val = sess.run([train_op, loss, summary, global_step, learning_rate])
                duration += time.time() - start_time

                if global_step_val % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0
                    print('=> %s: step %d, loss = %f (%.1f examples/sec)' % (
                        datetime.now(), global_step_val, loss_val, examples_per_sec))

                if global_step_val % num_steps_to_check != 0:
                    continue

                summary_writer.add_summary(summary_val, global_step=global_step_val)

                print('=> Evaluating on validation dataset...')
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'latest.ckpt'))
                accuracy = evalu.evaluate(path_to_latest_checkpoint_file, data.valid_dataset, data.valid_labels,
                                              global_step_val)
                print('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'model.ckpt'),
                                                         global_step=global_step_val)
                    print('=> Model saved to file: %s' % path_to_checkpoint_file)
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1

                print('=> patience = %d' % patience)
                if patience == 0:
                    break

            coord.request_stop()
            coord.join(threads)
            print('Finished')  



def main(_):
    path_to_pickle_file = os.path.join(FLAGS.data_dir, 'multi_crop.pickle')
#    path_to_pickle_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')
    path_to_train_log_dir = FLAGS.train_logdir
    path_to_restore_checkpoint_file = FLAGS.restore_checkpoint
    training_options = {
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'patience': FLAGS.patience,
        'decay_steps': FLAGS.decay_steps,
        'decay_rate': FLAGS.decay_rate
    }
    model_options = {
            'conv_kern': FLAGS.conv_kern, 
            'fil_1': FLAGS.fil_1, 
            'fil_2': FLAGS.fil_2, 
            'fil_3': FLAGS.fil_3, 
            'fil_4': FLAGS.fil_4, 
            'fil_5': FLAGS.fil_5, 
            'fil_fc': FLAGS.fil_fc,
            'fl_l': FLAGS.fl_l, 
            'fl_dig': FLAGS.fl_dig,
            'stride_odd': FLAGS.stride_odd, 
            'stride_even': FLAGS.stride_even,
            'mp': FLAGS.mp,
            'drop_rate': FLAGS.drop_rate, 
            'mu': FLAGS.mu, 
            'sigma': FLAGS.sigma
            }

    _run(path_to_pickle_file, path_to_train_log_dir, 
           path_to_restore_checkpoint_file,
           training_options, model_options)
    
if __name__ == '__main__':
    tf.app.run(main=main)