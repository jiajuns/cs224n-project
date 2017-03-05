from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging

import numpy as np
import tensorflow as tf
from qa_model import Encoder

logging.basicConfig(level=logging.INFO)

class BiLSTM_Encoder(Encoder):

    def _LSTM_cell(self, hidden_size):
        return tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)

    def Question_BiLSTM(self, inputs, masks, length):
        with tf.variable_scope("Question_BiLSTM") as scope:
            lstm_fw_cell = self._LSTM_cell(self.hidden_size)
            lstm_bw_cell = self._LSTM_cell(self.hidden_size)
            seq_len = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, inputs = inputs, sequence_length = seq_len, dtype=tf.float32
            )
            output = tf.concat(2, outputs)
            final_hidden_output = tf.reshape(output[:, -1, :], (-1, 2 * self.hidden_size, 1))
        return final_hidden_output

    def Context_BiLSTM(self, inputs, masks, length):
        with tf.variable_scope("Context_BiLSTM") as scope:
            lstm_fw_cell = self._LSTM_cell(self.hidden_size)
            lstm_bw_cell = self._LSTM_cell(self.hidden_size)
            seq_len = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, inputs = inputs, sequence_length = seq_len, dtype=tf.float32
            )
            hidden_outputs = tf.concat(2, outputs)
        return hidden_outputs

    def attention(self, y_q, y_c):
        with tf.variable_scope('attention') as scope:
            w_a = tf.get_variable("w_alpha", shape = (2 * self.hidden_size, 2 * self.hidden_size),
                initializer=tf.contrib.layers.xavier_initializer())
            #tf.get_variable_scope().reuse_variables()
            y_c_reshape = tf.reshape(y_c, (-1, 2 * self.hidden_size))
            temp_y = tf.reshape(tf.matmul(y_c_reshape, w_a), (-1, self.max_context_len, 2 * self.hidden_size))
            alpha = tf.matmul(temp_y, y_q)
            normalised_alpha = tf.nn.softmax(alpha)
            c_t = tf.matmul(tf.reshape(normalised_alpha, [-1, 1, self.max_context_len]), y_c)
            print(c_t)
            w_attention = tf.get_variable('w_attention', shape = (4 * self.hidden_size, self.hidden_size),
                initializer=tf.contrib.layers.xavier_initializer())
            test = tf.stack([c_t, tf.reshape(y_q, (-1, 1, 2 * self.hidden_size))])
            print(test)
            attention_hidden_outputs = tf.matmul(test, w_attention)
        return 

    def encode(self, context, question, context_mask, question_mask):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        yq = self.Question_BiLSTM(question, question_mask, self.max_question_len)
        yc = self.Context_BiLSTM(context, context_mask, self.max_context_len)
        return self.attention(yq, yc)


class Dummy_Encoder(Encoder):
    def LSTM(self, inputs, masks, length):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs = inputs, dtype = tf.float32)
        return outputs

    def encode(self, context, question, context_mask, question_mask):
        return self.LSTM(context, context_mask, self.max_context_len)