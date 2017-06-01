from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from palutils.util import check_input_checkpoint, restore_from_checkpoint


def store_weights(sess, output_path):
    for var in tf.trainable_variables():
        filename = '{}-{}'.format(output_path, var.name.replace(':', '-'))

        if var.name in ['Variable:0', 'Variable_2:0']:
            var = tf.transpose(var, perm=[3, 0, 1, 2])

        value = sess.run(var)

        with open(filename, b'w') as file_:
            value.tofile(file_)


def store_weights_from_checkpoint(input_checkpoint, output_path):
    check_input_checkpoint(input_checkpoint)

    with tf.Session() as sess:
        restore_from_checkpoint(sess, input_checkpoint)
        store_weights(sess, output_path)
