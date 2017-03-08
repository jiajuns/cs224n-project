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

    def decode(self):
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
            # attention (?, 1, 2h) ?
            # y_c (?, m, 2h)
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
            p_start = tf.nn.softmax(h_start)
            p_end = tf.nn.softmax(h_end)

        return p_start, p_end


class BiLSTM_Decoder(Decoder):
    def model_layer(self, context_mask, G):
        # G (?, m, 8h)
        with tf.variable_scope('model_layer'):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
            seq_len = tf.reduce_sum(tf.cast(context_mask, tf.int32), axis=1)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, inputs = G, sequence_length=seq_len, dtype=tf.float32
            )
            hidden_outputs = tf.concat(2, outputs)
        return hidden_outputs

    def output_layer(self, G, M):
        # M (?, m, 2h)
        # the softmax part is implemented together with loss function
        with tf.variable_scope('output_layer'):
            w_1 = tf.get_variable('w_start', shape=(10 * self.hidden_size, 1),
                initializer=tf.contrib.layers.xavier_initializer())
            w_2 = tf.get_variable('w_end', shape=(10 * self.hidden_size, 1),
                initializer=tf.contrib.layers.xavier_initializer())

            temp_1 = tf.concat(2, [G, M])  # (?, m, 10h)
            temp_1_reshape = tf.reshape(temp_1, shape=[-1, 10 * self.hidden_size])  # (?m, 10h)
            h_1 = tf.reshape(tf.matmul(temp_1_reshape, w_1), [-1, self.max_context_len]) # (?m, 10h) * (10h, 1) -> (?m, 1) -> (?, m)

            temp_2 = tf.concat(2, [G, M])  # (?, m, 10h)
            temp_2_reshape = tf.reshape(temp_2, shape=[-1, 10 * self.hidden_size])  # (?m, 10h)
            h_2 = tf.reshape(tf.matmul(temp_2_reshape, w_2), [-1, self.max_context_len]) # (?m, 10h) * (10h, 1) -> (?m, 1) -> (?, m)

            return h_1, h_2

    def decode(self, context_mask, G):
        M = self.model_layer(context_mask, G)
        h1, h2 = self.output_layer(G, M)
        return h1, h2