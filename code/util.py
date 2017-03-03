"""Utilities for loading and padding dataset"""
def load_and_preprocess_data(data_dir):
	
	# logger.info("Loading training data...")
 #    train = read_conll(args.data_train)
 #    logger.info("Done. Read %d sentences", len(train))
 #    logger.info("Loading dev data...")
 #    dev = read_conll(args.data_dev)
 #    logger.info("Done. Read %d sentences", len(dev))

	# load train data
	# train_context = read_data_from_file(data_dir + '/train.ids.context')
	# train_question = read_data_from_file(data_dir + '/train.ids.question')
	# train_span = read_data_from_file(data_dir + '/train.span')
	# train_data = vectorize(train_context, train_question, train_span)

	# load val data
	val_context = read_data_from_file(data_dir + '/val.ids.context')
	val_question = read_data_from_file(data_dir + '/val.ids.question')
	val_span = read_data_from_file(data_dir + '/val.span')
	val_data = vectorize(val_context, val_question, val_span)
	print val_data[0]


def read_data_from_file(dir):
	ret = []
	with open(dir, 'r') as file:
		for line in file:
			ids_list = [int(i) for i in line.strip().split(" ")]
			ret.append(ids_list)
	return ret

'''
vectorize dataset into [(context1, quesiton1, span1),(context2, quesiton2, span2),...]
'''
def vectorize(context, question, span):
	return(zip(context, question, span))

####### Unfinished####
def pad_sequences(data, max_length):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.

    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word is itself a list of
            @n_features features. For example, the sentence "Chris
            Manning is amazing" and labels "PER PER O O" would become
            ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
            the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
            is the list of labels. 
        max_length: the desired length for all input/output sequences.
    Returns:
        a new list of data points of the structure (sentence', labels', mask).
        Each of sentence', labels' and mask are of length @max_length.
        See the example above for more details.
    """
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * Config.n_features
    zero_label = 4 # corresponds to the 'O' tag

    for sentence, labels in data:
        ### YOUR CODE HERE (~4-6 lines)
        if len(sentence) >= max_length:
            new_sentence = sentence[:max_length]
            new_labels = labels[:max_length]
            mask = [True] * max_length
        else:
            new_sentence = sentence + [zero_vector] * (max_length - len(sentence))
            new_labels = labels + [zero_label] * (max_length - len(sentence))
            mask = [True] * len(sentence) + [False] * (max_length - len(sentence))
        ret.append((new_sentence, new_labels, mask))
        ### END YOUR CODE ###
    return ret

if __name__ == '__main__':
	print "Testing"
	load_and_preprocess_data('data/squad')

