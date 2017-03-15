from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import QASystem
from decoder import BiLSTM_Decoder as Decoder
from encoder import BiLSTM_Encoder as Encoder
from util import load_and_preprocess_data, load_embeddings,split_train_dev
from train import FLAGS
from evaluate import exact_match_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)

def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    overall_f1 = 0.
    overall_em = 0.
    minibatch_size = 100
    num_batches = int(len(dataset) / minibatch_size)
    #num_batches = 10
    for batch in range(0, num_batches):
        start = batch * minibatch_size
        print("batch {} out of {}".format(batch+1, num_batches))
        batch_f1 = 0.
        batch_em = 0.
        h_s, h_e = model.decode(sess, dataset[start:start + minibatch_size])
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
            # formulate questions and answers for computing the accuracy
            # question = sample_dataset[0][2]
            # question_mask = sample_dataset[0][3]
            # question_string = model.formulate_answer(question, rev_vocab, 0, len(question) - 1, mask = question_mask)
            predicted_answer = model.formulate_answer(context, rev_vocab, a_s, a_e)
            true_answer = model.formulate_answer(context, rev_vocab, a_s_true, a_e_true)
            f1 = f1_score(predicted_answer, true_answer)
            overall_f1 += f1
            batch_f1 += f1
            if exact_match_score(predicted_answer, true_answer):
                overall_em += 1
                batch_em += 1
        print("batch F1: {}".format(batch_f1/minibatch_size))
        print("batch EM: {}".format(batch_em/minibatch_size))
    print("overall F1: {}".format(overall_f1/(num_batches*minibatch_size)))
    print("overall EM: {}".format(overall_em/(num_batches*minibatch_size)))
    
def main(_):
    #======Fill the model name=============
    train_dir = "train/full_baseline_model_bilinear_150"
    #======================================
    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    
    # ========= Load Dataset =========
    train_data,val_data  = load_and_preprocess_data(FLAGS.data_dir, FLAGS.max_context_len, FLAGS.max_question_len, size = FLAGS.train_size)
    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    embedding = tf.constant(load_embeddings(embed_path), dtype = tf.float32)
    encoder = Encoder(FLAGS.state_size, FLAGS.max_context_len, FLAGS.max_question_len, FLAGS.embedding_size, FLAGS.summary_flag, FLAGS.reg_scale)
    decoder = Decoder(FLAGS.state_size, FLAGS.max_context_len, FLAGS.max_question_len, FLAGS.output_size, FLAGS.summary_flag, FLAGS.reg_scale)
    qa = QASystem(encoder, decoder, FLAGS, embedding, rev_vocab)

    with tf.Session() as sess:
        train_dir = get_normalized_train_dir(train_dir)
        qa = initialize_model(sess, qa, train_dir)
        generate_answers(sess, qa, val_data, rev_vocab)


if __name__ == "__main__":
  tf.app.run()
