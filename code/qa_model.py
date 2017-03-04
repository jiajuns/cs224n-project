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

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

class Encoder(object):
    def __init__(self, hidden_size, max_context_len, max_question_len, vocab_dim):
        self.hidden_size = hidden_size
        self.vocab_dim = vocab_dim
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len

    ## not ready
    def LSTM(self, inputs, masks, length):
        # Current data input shape: (batch_size, length, vocab_dim)
        # Required shape: 'length' tensors list of shape (batch_size, n_input)
        # Permuting batch_size and length
        # inputs = tf.transpose(inputs, [1, 0, 2])
        # # Reshaping to (length*batch_size, vocab_dim)
        # inputs = tf.reshape(inputs, [-1, self.vocab_dim])
        # # Split to get a list of 'length' tensors of shape (batch_size, n_input)
        # inputs = tf.split(0, length, inputs)
        # print(len(inputs))
        # print(inputs[0])
        #initial_state = tf.zeros_like((None, ))
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        #seq_len = tf.reduce_sum(tf.cast(masks, tf.float32))
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs = inputs, dtype = tf.float32)
        return outputs

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
        return self.LSTM(context, context_mask, self.max_context_len)


class Decoder(object):
    def __init__(self, hidden_size, max_context_len, max_question_len, output_size):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len

    def decode(self, x):
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
        with vs.variable_scope("decoder"):
            U = tf.get_variable("U", shape = (self.hidden_size, self.output_size),
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b2", initializer = tf.zeros((self.output_size,)))
            preds = tf.matmul(x[-1], U) + b
            print(preds)
            print(tf.reshape(preds, (-1, self.max_context_len, self.output_size)))
        return preds

class QASystem(object):
    def __init__(self, encoder, decoder, max_context_len, max_question_len, embeddings):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len
        self.pretrained_embeddings = embeddings
        self.vocab_dim = encoder.vocab_dim
        self.n_class = 3  # 3 output class: start, end, null

        # ==== set up placeholder tokens ========
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, max_context_len))
        self.question_placeholder = tf.placeholder(tf.int32, shape = (None, max_question_len))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape = (None, max_context_len))
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape = (None, max_question_len))
        self.span_placeholder = tf.placeholder(tf.bool, shape = (None, max_context_len))
        #self.dropout_placeholder = tf.placeholder(tf.float32, shape=(None))



        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            context_embeddings, question_embeddings = self.setup_embeddings()
            preds = self.setup_system(context_embeddings, question_embeddings)
            self.loss = self.setup_loss(preds)

        # ==== set up training/updating procedure ====
        pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            vec_embeddings = tf.get_variable("embeddings", initializer=self.pretrained_embeddings)
            context_batch_embeddings = tf.nn.embedding_lookup(vec_embeddings, self.context_placeholder)
            question_batch_embeddings = tf.nn.embedding_lookup(vec_embeddings, self.question_placeholder)
            context_embeddings = tf.reshape(context_batch_embeddings, 
                    (-1, self.max_context_len, self.vocab_dim))
            question_embeddings = tf.reshape(question_batch_embeddings, 
                    (-1, self.max_question_len, self.vocab_dim))
        return context_embeddings, question_embeddings

    def setup_system(self, context_embeddings, question_embeddings):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encoded_layer = self.encoder.encode(context_embeddings, question_embeddings, 
                        self.context_mask_placeholder, self.question_mask_placeholder)
        preds = self.decoder.decode(encoded_layer)
        return preds


    def setup_loss(self, preds):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            masked_pred = tf.boolean_mask(preds, self.context_mask_placeholder)
            masked_label = tf.boolean_mask(self.span_placeholder, self.context_mask_placeholder)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(masked_pred, masked_label))
        return loss

    def optimize(self, session, context, question, context_mask, question_mask, span):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {
                    self.context_placeholder: context, 
                    self.question_placeholder: question, 
                    self.context_mask_placeholder: context_mask, 
                    self.question_mask_placeholder: question_mask, 
                    self.span_placeholder: span
                    }
        output_feed = [self.context_placeholder]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        print(dataset)

