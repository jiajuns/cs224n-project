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

    # def LSTM(self, inputs, masks, length):
    #     # Current data input shape: (batch_size, length, vocab_dim)
    #     # Required shape: 'length' tensors list of shape (batch_size, n_input)
    #     # Permuting batch_size and length
    #     # inputs = tf.transpose(inputs, [1, 0, 2])
    #     # # Reshaping to (length*batch_size, vocab_dim)
    #     # inputs = tf.reshape(inputs, [-1, self.vocab_dim])
    #     # # Split to get a list of 'length' tensors of shape (batch_size, n_input)
    #     # inputs = tf.split(0, length, inputs)
    #     # print(len(inputs))
    #     # print(inputs[0])
    #     #initial_state = tf.zeros_like((None, ))
    #     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
    #     #seq_len = tf.reduce_sum(tf.cast(masks, tf.float32))
    #     outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs = inputs, dtype = tf.float32)
    #     return outputs

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
        preds = []

        with vs.variable_scope("decoder"):
            U = tf.get_variable("U", shape = (self.hidden_size, self.output_size),
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b2", initializer = tf.zeros((self.output_size,)))
            tf.get_variable_scope().reuse_variables()
            for time_step in xrange(self.max_context_len):
                temp_x = x[:, time_step, :]
                output = tf.matmul(temp_x, U) + b
                preds.append(output)
        return preds

class QASystem(object):
    def __init__(self, encoder, decoder, flags, embeddings, rev_vocab):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder
        self.max_context_len = flags.max_context_len
        self.max_question_len = flags.max_question_len
        self.pretrained_embeddings = embeddings
        self.vocab_dim = encoder.vocab_dim
        self.lr = flags.learning_rate
        self.n_epoch = flags.epochs
        self.batch_size = flags.batch_size
        self.rev_vocab = rev_vocab

        # ==== set up placeholder tokens ========
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_context_len))
        self.question_placeholder = tf.placeholder(tf.int32, shape = (None, self.max_question_len))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape = (None, self.max_context_len))
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape = (None, self.max_question_len))
        self.span_placeholder = tf.placeholder(tf.int32, shape = (None, self.max_context_len))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(None))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            context_embeddings, question_embeddings = self.setup_embeddings()
            self.preds = self.setup_system(context_embeddings, question_embeddings)
            self.loss, self.masked_pred = self.setup_loss(self.preds)

        # ==== set up training/updating procedure ====
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

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
        yq, yc, attention = self.encoder.encode(context_embeddings, question_embeddings,
                        self.context_mask_placeholder, self.question_mask_placeholder)
        preds = self.decoder.decode(yc, attention)
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
        return loss, masked_pred

    def create_feed_dict(self, train_batch, dropout=1, test_flag = False):
        feed_dict = {
            self.context_placeholder: train_batch[0],
            self.question_placeholder: train_batch[2],
            self.context_mask_placeholder: train_batch[1],
            self.question_mask_placeholder: train_batch[3],
        }
        if len(train_batch) == 6:
            feed_dict[self.span_placeholder] = train_batch[4]
        return feed_dict

    def optimize(self, session, train_batch):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = self.create_feed_dict(train_batch)
        output_feed = [self.train_op, self.loss]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def run_epoch(self, session, train_examples, dev_examples):
        # train_examples, dev_examples
        #     [(context, context_mask, question, question_mask, span_sparse, span_sparse)]
        prog = Progbar(target=int(len(train_examples) / self.batch_size))
        for i, batch in enumerate(minibatches(train_examples, self.batch_size)):
            outputs = self.optimize(session, batch)
            prog.update(i + 1, [("train loss", outputs[1])])
            # if self.report: self.report.log_train_loss(loss)
        print("")

        logging.info("Evaluating on development data")
        validate_cost = self.test(session, dev_examples)
        return validate_cost

    def test(self, session, dev_example):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """

        # fill in this feed_dictionary like:
        unzipped_dev_example = zip(*dev_example)
        input_feed = self.create_feed_dict(unzipped_dev_example)
        output_feed = [self.loss]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]


    ###### Under Work! ##########
    def decode(self, session, dev_example):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        unzipped_dev_example = zip(*dev_example)
        input_feed = self.create_feed_dict(unzipped_dev_example)
        output_feed = [self.masked_pred]
        masked_pred = session.run(output_feed, input_feed)
        #print(masked_pred)
        yp = masked_pred[0]
        return yp

    def answer(self, session, test_x):

        yp = self.decode(session, test_x)
        answer_index = np.argmax(yp, axis = 1)
        #print(answer_index)
        return answer_index

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        return self.test(sess, valid_dataset)

    def formulate_answer(self, context, rev_vocab, start, end):
        answer = ''
        for i in range(start, end+1):
            answer +=  rev_vocab[context[i]]
            answer += ' '
        return answer


    def evaluate_answer(self, session, dataset, rev_vocab, sample=100, log=False):
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

        for i in range(sample):
            sample_dataset = [dataset[i]] ## batch size = 1, keep same format after indexing
            (a_s, a_e) = self.answer(session, sample_dataset)

            (a_s_true, a_e_true) = sample_dataset[0][5]
            context = sample_dataset[0][0]
            question = sample_dataset[0][2]
            print(self.formulate_answer(question, rev_vocab, 0, len(question)))
            predicted_answer = self.formulate_answer(context, rev_vocab, a_s, a_e)
            print(a_s, a_e)
            print("predicted answer: {}".format(predicted_answer))
            true_answer = self.formulate_answer(context, rev_vocab, a_s_true, a_e_true)
            print("true answer: {}".format(true_answer))
            print(f1_score(predicted_answer, true_answer))
            #print(exact_match_score(predicted_answer, true_answer))
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

        self.saver = tf.train.Saver()
        train_examples, dev_examples = split_train_dev(dataset)

        best_score = 0
        for epoch in range(self.n_epoch):
            print("Epoch {:} out of {:}".format(epoch + 1, self.n_epoch))
            dev_score = self.run_epoch(session, train_examples, dev_examples)
            print("Dev Cost: {}".format(dev_score))
            f1, em = self.evaluate_answer(session, dev_examples, self.rev_vocab)
            if dev_score > best_score:
                best_score = dev_score
                print("New best dev score! Saving model in {}".format(train_dir))
                self.saver.save(session, train_dir)

        return best_score
