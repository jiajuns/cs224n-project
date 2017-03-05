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

        """
        h_hat ==> (batch_size,1，2h)
        y_con ==> (batch_size,m,2h)
        """

        n_hidden_units = h_hat.shape[2]
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, y_con, initial_state=h_hat, time_major=False)

        return outputs
