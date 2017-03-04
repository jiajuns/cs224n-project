from __future__ import print_function
from six.moves import xrange  # for python 3 user
import time, logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def load_and_preprocess_data(data_dir, max_context_len = 2834, max_question_len = 214, debug = True):
    """Utilities for loading and padding dataset"""
    start = time.time()
    logger.info("Loading training data...")
    if debug:
        logger.info("DEBUG Mode")
        train_context = read_data_from_file(data_dir + '/toy.ids.context')
        train_question = read_data_from_file(data_dir + '/toy.ids.question')
        train_span = read_data_from_file(data_dir + '/toy.span')
    else:
        logger.info("Training Mode")
        train_context = read_data_from_file(data_dir + '/train.ids.context')
        train_question = read_data_from_file(data_dir + '/train.ids.question')
        train_span = read_data_from_file(data_dir + '/train.span')
    train_context_padded, train_context_mask = pad_sequence(train_context, max_context_len)
    train_question_padded, train_question_mask = pad_sequence(train_question, max_question_len)
    train_span_processed = preprocess_span(train_span, train_context_padded)
    train_data = vectorize(train_context_padded, train_context_mask,
                        train_question_padded, train_question_mask, train_span_processed)
    logger.info("Done. Read %d sentences", len(train_data))
    # logger.info("Loading validation data...")
    # val_context = read_data_from_file(data_dir + '/val.ids.context')
    # val_question = read_data_from_file(data_dir + '/val.ids.question')
    # val_span = read_data_from_file(data_dir + '/val.span')
    # val_context_padded, val_context_mask = pad_sequence(val_context, max_context_len)
    # val_question_padded, val_question_mask = pad_sequence(val_question, max_question_len)
    # val_span_processed = preprocess_span(val_span, val_context_padded)
    # val_data = vectorize(val_context_padded, val_context_mask,
    #                     val_question_padded, val_question_mask, val_span_processed)
    # logger.info("Done. Read %d sentences", len(val_data))
    # logger.info("Data Loaded. Took %d seconds", time.time()-start)
    return train_data,1
    #return train_data, val_data

def read_data_from_file(dir):
    ret = []
    with open(dir, 'r') as file:
        for line in file:
            ids_list = [int(i) for i in line.strip().split(" ")]
            ret.append(ids_list)
    return ret

def vectorize(context, context_mask, question, question_mask, span):
    '''
    Vectorize dataset into
    [(context1, context_mask1, quesiton1, question_mask1, span1),
    (context2, context_mask2, quesiton2, question_mask2, span2),...]
    '''
    return(zip(context, context_mask, question, question_mask, span))

def preprocess_span(span_vector, context):
    new_span_vector = []
    for i in xrange(len(span_vector)):
        new_span = [0] * len(context[i])
        try:
            new_span[span_vector[i][0]] = 1
        except:
            print(i, span_vector[i], len(span_vector), len(context[i]))
        new_span[span_vector[i][1]] = 2
        new_span_vector.append(new_span)
    return new_span_vector

def pad_sequence(data, max_length):
    """
    Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.
    """
    padded_sequence = []
    masks = []
    # Use this zero vector when padding sequences.
    zero_label = 0 # corresponds to the 'O' tag

    for sentence in data:
        if len(sentence) >= max_length:
            new_sentence = sentence[:max_length]
            mask = [True] * max_length
        else:
            new_sentence = sentence + [zero_label] * (max_length - len(sentence))
            mask = [True] * len(sentence) + [False] * (max_length - len(sentence))

        padded_sequence.append(new_sentence)
        masks.append(mask)
    return padded_sequence, masks

def load_embeddings(dir):
    return np.load(dir)['glove']

if __name__ == '__main__':
    print("Testing")
    load_and_preprocess_data('data/squad')

