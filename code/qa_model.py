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
        self.n_epoch = flags.epochs
        self.batch_size = flags.batch_size
        self.rev_vocab = rev_vocab
        self.dropout = flags.dropout
        self.summaries_dir = flags.summaries_dir
        self.summary_flag = flags.summary_flag
        self.max_grad_norm = flags.max_grad_norm
        self.reg_scale = flags.reg_scale
        self.base_lr = flags.learning_rate
        self.decay_number = flags.decay_number
        self.model_name = flags.model_name
        self.train_loss_log = flags.train_dir + "/" + "train_loss.csv"
        self.dev_loss_log = flags.train_dir + "/" + "dev_loss.csv"

        # ==== set up placeholder tokens ========
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_context_len))
        self.question_placeholder = tf.placeholder(tf.int32, shape = (None, self.max_question_len))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape = (None, self.max_context_len))
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape = (None, self.max_question_len))
        self.start_span_placeholder = tf.placeholder(tf.int32, shape = (None, self.max_context_len))
        self.end_span_placeholder = tf.placeholder(tf.int32, shape = (None, self.max_context_len))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(None))
        self.global_batch_num_placeholder = tf.placeholder(tf.int32, shape=(None))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            context_embeddings, question_embeddings = self.setup_embeddings()
            self.h_s, self.h_e = self.setup_system(context_embeddings, question_embeddings)
            self.loss, self.masked_h_s, self.masked_h_e = self.setup_loss(self.h_s, self.h_e)
            # computing learning rates
            self.learning_rate = tf.train.exponential_decay(
              self.base_lr,                               # Base learning rate.
              self.global_batch_num_placeholder,          # Current total batch number
              self.decay_number,                          # decay every 50 batch
              0.99,                                       # Decay rate
              staircase = True)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # ==== set up training/updating procedure ====
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            grads = [gv[0] for gv in grads_and_vars]
            variables = [gv[1] for gv in grads_and_vars]
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.train_op = self.optimizer.apply_gradients(zip(grads, variables))
            if self.summary_flag:
                tf.summary.scalar('cross_entropy', self.loss)
                self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            vec_embeddings = tf.get_variable("embeddings", initializer=self.pretrained_embeddings, trainable=False)
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
                        self.context_mask_placeholder, self.question_mask_placeholder,
                        self.dropout_placeholder)
        h_s, h_e = self.decoder.decode(self.context_mask_placeholder, self.dropout_placeholder, attention)
        return h_s, h_e

    def setup_loss(self, h_s, h_e):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            masked_h_s = tf.boolean_mask(h_s, self.context_mask_placeholder)
            masked_h_e = tf.boolean_mask(h_e, self.context_mask_placeholder)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h_s, self.start_span_placeholder) +
                   tf.nn.softmax_cross_entropy_with_logits(h_e, self.end_span_placeholder))
            # reg_vars = [tf_var for tf_var in tf.trainable_variables() if "Bias" not in tf_var.name]
            # tf.contrib.layers.apply_regularization(
            #     tf.contrib.layers.l2_regularizer(self.reg_scale), reg_vars)
            # reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # total_loss = loss + sum(reg_loss)
            total_loss = loss
        return total_loss, masked_h_s, masked_h_e

    def create_feed_dict(self, train_batch, dropout, global_batch_num = 0):
        feed_dict = {
            self.context_placeholder: train_batch[0],
            self.question_placeholder: train_batch[2],
            self.context_mask_placeholder: train_batch[1],
            self.question_mask_placeholder: train_batch[3],
            self.dropout_placeholder: dropout,
            self.global_batch_num_placeholder: global_batch_num
        }
        if len(train_batch) == 7:
            feed_dict[self.start_span_placeholder] = train_batch[4]
            feed_dict[self.end_span_placeholder] = train_batch[5]
        return feed_dict

    def optimize(self, session, train_batch, global_batch_num):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = self.create_feed_dict(train_batch, 1 - self.dropout, global_batch_num)
        if self.summary_flag:
            output_feed = [self.train_op, self.loss, self.merged, self.learning_rate]
            _, loss, summary, current_lr = session.run(output_feed, input_feed)
        else:
            output_feed = [self.train_op, self.loss, self.learning_rate]
            _, loss, current_lr = session.run(output_feed, input_feed)
            summary = None
        return loss, summary, current_lr

    def run_epoch(self, session, train_examples, dev_examples, epoch_num, train_log):
        num_batches = int(len(train_examples) / self.batch_size)
        prog = Progbar(target=num_batches)
        for i, batch in enumerate(minibatches(train_examples, self.batch_size)):
            loss, summary, current_lr = self.optimize(session, batch, global_batch_num = epoch_num * num_batches + i)
            # logging format (epoch,loss)
            train_log.write("{},{}\n".format(epoch_num + 1, loss))
            prog.update(i + 1, exact = [("train loss", loss), ("current LR", current_lr)])
            if self.summary_flag:
                self.train_writer.add_summary(summary, i)
        print("")
        logging.info("Evaluating on development data")
        validate_cost = self.test(session, dev_examples)
        return validate_cost

    def test(self, session, dev_examples):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """

        num_batches = int(len(dev_examples) / self.batch_size)
        prog = Progbar(target=num_batches)
        total_cost = 0
        for i, batch in enumerate(minibatches(dev_examples, self.batch_size)):
            input_feed = self.create_feed_dict(batch, dropout = 1)
            output_feed = [self.loss]
            outputs = session.run(output_feed, input_feed)
            prog.update(i + 1, exact = [("dev loss", outputs[0])])
            total_cost += outputs[0]
        print("")

        return total_cost/(i + 1)

    def decode(self, session, dev_example):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        unzipped_dev_example = list(zip(*dev_example))
        input_feed = self.create_feed_dict(unzipped_dev_example[0:4], dropout = 1)
        output_feed = [self.h_s, self.h_e]
        outputs = session.run(output_feed, input_feed)
        h_s = outputs[0]
        h_e = outputs[1]
        return h_s, h_e

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

    def formulate_answer(self, context, rev_vocab, start, end, mask = None):
        answer = ''
        for i in range(start, end + 1):
            if i < len(context):
                if mask is None:
                    answer +=  rev_vocab[context[i]]
                    answer += ' '
                else:
                    if mask[i]:
                        answer +=  rev_vocab[context[i]]
                        answer += ' '
        return answer

    def evaluate_answer(self, session, dataset, rev_vocab, sample=300, log=False):
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
        sample = min(sample, len(dataset))
        overall_f1 = 0.
        overall_em = 0.
        minibatch_size = 100
        num_batches = int(sample / minibatch_size)
        for batch in range(0, num_batches):
            start = batch * minibatch_size
            end = min(len(dataset), start + minibatch_size)
            h_s, h_e = self.decode(session, dataset[start:end])
            for i in range(minibatch_size):
                a_s = np.argmax(h_s[i])
                a_e = np.argmax(h_e[i])
                if a_s > a_e:
                    k = a_e
                    a_e = a_s
                    a_s = k

                sample_dataset = dataset[start + i]
                context = sample_dataset[0]
                (a_s_true, a_e_true) = sample_dataset[6]
                predicted_answer = self.formulate_answer(context, rev_vocab, a_s, a_e)
                true_answer = self.formulate_answer(context, rev_vocab, a_s_true, a_e_true)
                f1 = f1_score(predicted_answer, true_answer)
                overall_f1 += f1
                if exact_match_score(predicted_answer, true_answer):
                    overall_em += 1

        average_f1 = overall_f1/sample
        overall_em = overall_em/sample
        logging.info("F1: {}, EM: {}, for {} samples\n".format(average_f1, overall_em, sample))
        return overall_f1, overall_em

    def train(self, session, train_examples, dev_examples, train_dir):
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

        if self.summary_flag:
            self.train_writer = tf.summary.FileWriter(self.summaries_dir + '/train', session.graph)

        logging.info("Train Loss File: {}".format(self.train_loss_log))
        logging.info("Dev Loss File: {}".format(self.dev_loss_log))
        best_score = 100000
        train_log = open(self.train_loss_log, "w")
        dev_log = open(self.dev_loss_log, "w")
        for epoch in range(self.n_epoch):
            print("Epoch {:} out of {:}".format(epoch + 1, self.n_epoch))
            dev_score = self.run_epoch(session, train_examples, dev_examples, epoch, train_log)
            dev_log.write("{}.{}\n".format(epoch + 1, dev_score))
            logging.info("Average Dev Cost: {}".format(dev_score))
            logging.info("train F1 & EM")
            f1, em = self.evaluate_answer(session, train_examples, self.rev_vocab, log = True)
            logging.info("Dev F1 & EM")
            f1, em = self.evaluate_answer(session, dev_examples, self.rev_vocab, log = True)
            if dev_score < best_score:
                best_score = dev_score
                print("New best dev score! Saving model in {}".format(train_dir + "/" + self.model_name))
                self.saver.save(session, train_dir + "/" + self.model_name)

        return best_score
