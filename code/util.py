import time, logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
"""Utilities for loading and padding dataset"""
def load_and_preprocess_data(data_dir, max_context_len = 2834, max_question_len = 214):
	start = time.time()
	logger.info("Loading training data...")
	train_context = read_data_from_file(data_dir + '/train.ids.context')
	train_question = read_data_from_file(data_dir + '/train.ids.question')
	train_span = read_data_from_file(data_dir + '/train.span')
	train_context_padded, train_context_mask = pad_sequence(train_context, max_context_len)
	train_question_padded, train_question_mask = pad_sequence(train_question, max_question_len)
	train_data = vectorize(train_context_padded, train_context_mask,
						train_question_padded, train_question_mask, train_span)
	logger.info("Done. Read %d sentences", len(train_data))
	logger.info("Loading validation data...")
	val_context = read_data_from_file(data_dir + '/val.ids.context')
	val_question = read_data_from_file(data_dir + '/val.ids.question')
	val_span = read_data_from_file(data_dir + '/val.span')
	val_context_padded, val_context_mask = pad_sequence(val_context, max_context_len)
	val_question_padded, val_question_mask = pad_sequence(val_question, max_question_len)
	val_data = vectorize(val_context_padded, val_context_mask, 
						val_question_padded, val_question_mask, val_span)
	logger.info("Done. Read %d sentences", len(val_data))
	logger.info("Data Loaded. Took %d seconds", time.time()-start)
	return train_data, val_data

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

if __name__ == '__main__':
	print "Testing"
	load_and_preprocess_data('data/squad')

