from __future__ import print_function
from six.moves import xrange  # for python 3 user
import time, logging
import sys
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
    start_span_vector, end_span_vector = preprocess_span(train_span, train_context_padded)
    train_data = vectorize(train_context_padded, train_context_mask,
                        train_question_padded, train_question_mask, start_span_vector, end_span_vector, train_span)
    logger.info("Done. Read %d sentences", len(train_data))
    # logger.info("Loading validation data...")
    # val_context = read_data_from_file(data_dir + '/val.ids.context')
    # val_question = read_data_from_file(data_dir + '/val.ids.question')
    # val_span = read_data_from_file(data_dir + '/val.span')
    # val_context_padded, val_context_mask = pad_sequence(val_context, max_context_len)
    # val_question_padded, val_question_mask = pad_sequence(val_question, max_question_len)
    # val_span_processed = preprocess_span(val_span, val_context_padded)
    # val_data = vectorize(val_context_padded, val_context_mask,
    #                     val_question_padded, val_question_mask, val_span_processed, val_span)
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

def vectorize(context, context_mask, question, question_mask, start_span, end_span, span):
    '''
    Vectorize dataset into
    [(context1, context_mask1, quesiton1, question_mask1, span1),
    (context2, context_mask2, quesiton2, question_mask2, span2),...]
    '''
    return list(zip(context, context_mask, question, question_mask, start_span, end_span, span))

def preprocess_span(span_vector, context):
    start_span_vector = []
    end_span_vector = []
    for i in xrange(len(span_vector)):
        start_span = [0] * len(context[i])
        end_span = [0] * len(context[i])
        if span_vector[i][0] < len(context[i]):
                start_span[span_vector[i][0]] = 1
        if span_vector[i][1] < len(context[i]):
                end_span[span_vector[i][0]] = 1
        start_span_vector.append(start_span)
        end_span_vector.append(end_span)
    return start_span_vector, end_span_vector

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

def split_train_dev(data, split=0.9):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    train_len = int(data_size * split)
    train_index = [indices[i] for i in xrange(train_len)]
    dev_index = [indices[i] for i in xrange(train_len, data_size)]

    train_data = data[train_index] if type(data) is np.ndarray else [data[i] for i in train_index]
    dev_data = data[dev_index] if type(data) is np.ndarray else [data[i] for i in dev_index]

    return train_data, dev_data

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)

def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.seed(1)
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)

if __name__ == '__main__':
    print("Testing")
    load_and_preprocess_data('data/squad')
