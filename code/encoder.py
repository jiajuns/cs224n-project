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
            # outputs = tf.pack(outputs, axis=1)
            outputs = tf.concat(2, outputs)
            print('question_outputs_concat:', outputs)
            final_hidden_output = outputs[:,-1,:]
            print('final_hidden', final_hidden_output)
        return final_hidden_output

    def Context_BiLSTM(self, inputs, masks, length):
        with tf.variable_scope("Context_BiLSTM") as scope:
            lstm_fw_cell = self._LSTM_cell(self.hidden_size)
            lstm_bw_cell = self._LSTM_cell(self.hidden_size)
            seq_len = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, inputs = inputs, sequence_length = seq_len, dtype=tf.float32
            )
            # outputs = tf.pack(outputs, axis=1)
            outputs = tf.concat(2, outputs)
            print('Context_outputs_packed', outputs)
        return outputs

    def attention(self, output_q, output_c):
        with tf.variable_scope('attention') as scope:
            w_a = tf.get_variable("w_alpha", shape = (2 * self.hidden_size, 2 * self.hidden_size),
                initializer=tf.contrib.layers.xavier_initializer())
            alpha = tf.matmul(tf.matmul(output_c, w_a), tf.transpose(output_q))
            normalised_alpha = tf.nn.softmax(alpha)
            h_c = tf.matmul(tf.transpose(normalised_alpha), output_c)
            w = tf.get_variable('w_attention', shape = (4 * self.hidden_size, self.hidden_size),
                initializer=tf.contrib.layers.xavier_initializer())
        return tf.matmul(tf.stack([h_c, output_q]), w)

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