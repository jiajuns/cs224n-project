from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, json, argparse

import tensorflow as tf

#from qa_model import Encoder, QASystem, Decoder
from qa_model import QASystem
from decoder import BiLSTM_Decoder as Decoder
from encoder import BiLSTM_Encoder as Encoder
from os.path import join as pjoin
from util import load_and_preprocess_data, load_embeddings
import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_grad_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.25, "Fraction of units randomly dropped on non-recurrent connections.")
<<<<<<< HEAD
tf.app.flags.DEFINE_integer("train_size", 300, "Size of the training data")
tf.app.flags.DEFINE_integer("batch_size", 30, "Batch size to use during training.")
=======
tf.app.flags.DEFINE_integer("train_size", 20000, "Size of the training data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
>>>>>>> f77f69505c563ab072ccf4072a0cbef1a46b320b
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 2, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("prediction_log", "log/prediction", "Path to prediction file")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_integer("max_context_len", 300, "max length of the context input")
tf.app.flags.DEFINE_integer("max_question_len", 100, "max length of the question input")
tf.app.flags.DEFINE_string('summaries_dir', 'summary/', 'tensorboard summary dir')
<<<<<<< HEAD
tf.app.flags.DEFINE_bool('summary_flag', False, 'if true log summary')
tf.app.flags.DEFINE_float('reg_scale', 0.00001, 'regularization scale')
=======
tf.app.flags.DEFINE_bool('summary_flag', True, 'if true log summary')
tf.app.flags.DEFINE_float('reg_scale', 0.00005, 'regularization scale')
>>>>>>> f77f69505c563ab072ccf4072a0cbef1a46b320b
FLAGS = tf.app.flags.FLAGS


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

def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    train_data, val_data = load_and_preprocess_data(FLAGS.data_dir, FLAGS.max_context_len, FLAGS.max_question_len, size = FLAGS.train_size)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    embedding = tf.constant(load_embeddings(embed_path), dtype = tf.float32)
    encoder = Encoder(FLAGS.state_size, FLAGS.max_context_len, FLAGS.max_question_len, FLAGS.embedding_size, FLAGS.summary_flag, FLAGS.reg_scale)
    decoder = Decoder(FLAGS.state_size, FLAGS.max_context_len, FLAGS.max_question_len, FLAGS.output_size, FLAGS.summary_flag, FLAGS.reg_scale)

    qa = QASystem(encoder, decoder, FLAGS, embedding, rev_vocab)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, train_data, save_train_dir)

        #qa.evaluate_answer(sess, train_data, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
