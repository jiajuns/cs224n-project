from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
from util import Progbar, minibatches, split_train_dev

class Decoder(object):
    def __init__(self, hidden_size, max_context_len, max_question_len, output_size):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len

    def decode(self, h_hat, y_con):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        raise NotImplementedError("Each Model must re-implement this method.")


class LSTM_Decorder(Decoder):

    def LSTM(self, inputs):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(2 * self.hidden_size, forget_bias=1.0)
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs = inputs, dtype = tf.float32)
        return outputs

    def decode(self, y_c, attention):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        with vs.variable_scope('decode'):
            context_with_attention = attention * y_c
            h = self.LSTM(context_with_attention)     # (?, m, 2h)
            h = tf.reshape(h, shape=(-1, 2 * self.hidden_size))

            w_start = tf.get_variable('w_start', shape = (2 * self.hidden_size, 1),
                initializer=tf.contrib.layers.xavier_initializer())
            b_start = tf.get_variable('b_start', shape= (1))

            w_end = tf.get_variable('w_end', shape = (2 * self.hidden_size, 1),
                initializer=tf.contrib.layers.xavier_initializer())
            b_end = tf.get_variable('b_end', shape= (1))

            delta_start = tf.reshape(tf.matmul(h, w_start), shape=(-1, self.max_context_len))
            delta_end = tf.reshape(tf.matmul(h, w_start), shape=(-1, self.max_context_len))

            h_start = tf.nn.tanh(delta_start + b_start)
            h_end = tf.nn.tanh(delta_end + b_end)

        return h_start, h_end


