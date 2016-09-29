"""Convert tensorflow model into numpy matrices."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import argparse

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn.translate import seq2seq_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "-ckpt",
    "--ckpt_path",
    help="path tp the tensorflow checkpoints",
    required=True
)
parser.add_argument(
    "-opath",
    "--output_path",
    help="path to write the output variables",
    required=True
)

args = parser.parse_args()

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

with tf.device('/cpu:0'):
    ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    model = seq2seq_model.Seq2SeqModel(
        40000,
        40000,
        _buckets,
        1024,
        3,
        5,
        64,
        0.5,
        0.99,
        forward_only=True,
        use_lstm=True
    )
    model.saver.restore(sess, ckpt.model_checkpoint_path)
    for var in tf.trainable_variables():
        print(var.name, var.get_shape())
        if not var.name.startswith('embedding_attention_seq2seq'):
            continue
        if not var.name.split('/')[1] == 'RNN':
            continue
        if var.name.split('/')[2] == 'EmbeddingWrapper':
            print('Saving Embedding ...')
            tf_variable = sess.run(var)
            np.save(
                open('%s/%s.npy' % (args.output_path, 'src_embedding'), 'wb'),
                tf_variable
            )
        if 'BasicLSTMCell' in var.name and 'decoder' not in var.name:
            print('Saving LSTM ...')
            cell_number = var.name.split('/')[3][-1]
            tf_variable = sess.run(var)
            if 'Matrix' in var.name.split('/')[-1]:
                print('Saving LSTM W, U ...')
                tf_variable_W = tf_variable[:1024]
                tf_variable_U = tf_variable_W
                np.save(
                    open('%s/%s.npy' % (args.output_path, 'lstm_W_%s' % (cell_number)), 'wb'),
                    tf_variable_W
                )
                np.save(
                    open('%s/%s.npy' % (args.output_path, 'lstm_U_%s' % (cell_number)), 'wb'),
                    tf_variable_U
                )
            elif 'Bias' in var.name.split('/')[-1]:
                print('Saving LSTM B ...')
                np.save(
                    open('%s/%s.npy' % (args.output_path, 'lstm_b_%s' % (cell_number)), 'wb'),
                    tf_variable
                )
print('Finished saving model ...')
