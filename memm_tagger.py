#################################################
# ================ DO NOT MODIFY ================
#################################################
import sys
import math
import numpy as np
from collections import Counter
from sklearn import linear_model
from scipy import sparse
import pickle


# Dictionary to store indices for each feature
feature_vocab = {}

# Dictionary to store the indices of each POS tag
label_vocab = {}

#################################################
# ============== IMPLEMENT BELOW ================
#################################################

# Control variables, use for debugging and testing
verbose = True
use_greedy = False

####################################################################################
# FEEL FREE TO CHANGE MINIMUM_FEAT_COUNT AND L2_REGULARIZATION_COEFFICIENT IF NEEDED
####################################################################################

# Minimum number of observations for a feature to be included in model
MINIMUM_FEAT_COUNT = 2
# L2 regularization strength; range is (0,infinity), with stronger regularization -> 0 (in this package)
L2_REGULARIZATION_STRENGTH = 1e0

ABLATE = None
NEW_FEATURES = ['FIRST_WORD_IN_SEQUENCE', 'PREV_WORD_', 'PREV_BIGRAM_', 'NEXT_WORD_', 'NEXT_BIGRAM_', 'TRIGRAM_', 'NUMBER', 'SYMBOL_', 'CAPITALIZED', 'SUFFIX_']

SYMBOLS = "`~!@#$%^&*()_+-=\][|}{';\":/.,?><"
SUFFIXES_2 = ['ly', 'ty', 'er', 'ed', 'al', 'ic', 'en', 'cy']
SUFFIXES_3 = ['ful', 'ize', 'ing', 'dom', 'age', 'sis', 'ism', 'ity', 'ant', 'ily', 'ely', 'ive', 'ble', 'ous', 'ish', 'ian', 'ist', 'ize', 'ise', 'yse', 'ate', 'ify', 'ive', 'ial', 'est']
SUFFIXES_4 = ['ment', 'ness', 'sion', 'tion', 'ance', 'hood', 'ship', 'cess', 'less', 'like', 'some', 'fine']
SUFFIXES_5 = ['worthy']
SUFFIX_GROUPS = {2: SUFFIXES_2, 3: SUFFIXES_3, 4: SUFFIXES_4, 5: SUFFIXES_5}

# load up any external resources here
def initialize():
    """
        :return (dictionary data - any format that you wish, which
        can be reused in get_features below)
    """
    data = {}
    return data


def get_features(index, sequence, tag_index_1, data):
    """
        :params
        index: the index of the current word in the sequence to featurize

        sequence: the sequence of words for the entire sentence

        tag_index_1: gives you the POS
        tag for the previous word.

        data: the data you have built in initialize()
        to enrich your feature representation. Use data as you see fit.

        :return (feature dictionary)
        features are in the form of {feature_name: feature_value}
        Calculate the values of each feature for a given
        word in a sequence.

        The current implementation returns the following as features:
        the current word, the tag of the previous word, and whether an
        index the the last in the sequence.

    """
    features = {}

    features["UNIGRAM_%s" % sequence[index].lower()] = 1
    features["PREVIOUS_TAG_%s" % tag_index_1] = 1

    if index == len(sequence) - 1:
        features["LAST_WORD_IN_SEQUENCE"] = 1

    """
        Create your own features here
    """
    if index == 0:
        features["FIRST_WORD_IN_SEQUENCE"] = 1
    else:
        features["PREV_WORD_%s" % sequence[index - 1].lower()] = 1
        features["PREV_BIGRAM_%s_%s" % (sequence[index - 1].lower(), sequence[index].lower())] = 1

    if index < len(sequence) - 1:
        features["NEXT_WORD_%s" % sequence[index + 1].lower()] = 1
        features["NEXT_BIGRAM_%s_%s" % (sequence[index].lower(), sequence[index + 1].lower())] = 1

    if index > 0 and index < len(sequence) - 1:
        features["TRIGRAM_%s_%s_%s" % (sequence[index - 1].lower(), sequence[index].lower(), sequence[index + 1].lower())] = 1

    if sequence[index].replace(".", "").isdigit():
        features["NUMBER"] = 1
    if sequence[index] in SYMBOLS:
        features["SYMBOL_%s" % sequence[index]] = 1
    if sequence[index].lower() != sequence[index]:
        # if there exists capitalized letters
        features["CAPITALIZED"] = 1
    for suffix_len, suffixes in SUFFIX_GROUPS.items():
        if len(sequence[index]) > suffix_len + 2 and sequence[index][-suffix_len:] in suffixes:
            suffix = sequence[index][-suffix_len:]
            features["SUFFIX_%s" % suffix] = 1

    if ABLATE != None:
        ablation_prefix = NEW_FEATURES[ABLATE]
        remove_features = []
        for f in features:
            if f.startswith(ablation_prefix):
                remove_features.append(f)
        for f in remove_features:
            features.pop(f)

    return features


def viterbi_decode(Y_pred):
    """
        :return
        list of POS tag indices, defined in label_vocab,
        in order of sequence

        :param
        Y_pred: Tensor of shape N * M * L, where
        N is the number of words in the current sequence,
        L is the number of POS tag labels
        M = L + 1, where M is the number of possible previous labels
        which includes L + "START"

        M_{x,y,z} is the probability of a tag (label_vocab[current_tag] = z)
        given its current position x in the sequence and
        its previous tag (label_vocab[previous_tag] = y)

        Consider the sentence - "I have a dog". Here, N = 4.
        Assume that there are only 3 possible tags: {PRP, VBD, NN}
        M_{0, 3, 2} would give you the probability of "I" being a "NN" if
        it is preceded by "START". "START" is the last index of all labels,
        and in our example denoted by 3.
    """
    # cur = label_vocab["START"]
    cur = Y_pred[0].shape[0] - 1
    (N, M, L) = Y_pred.shape

    # list of POS tag indices to be returned
    path = []

    """
        Type your code below

        Hints:
        step 1. construct a Viterbi matrix to store the
        intermediate joint probabilities
        step 2. construct a matrix to store the backpointers
        to retrieve the path
        step 3. decode and return the path from the Viterbi matrix
        Note: Probabilities get smaller as you multiply them, and may
        vanish. Remember how we have treated chained probabilities
        this entire class.

    """
    # (START + number of words) * (START + number of labels)
    viterbi = np.zeros((N, L))
    backpointers = np.zeros((N, L), dtype=np.int32)

    # for first word, backpointer should point to START
    start = cur
    backpointers[0,:] = start
    viterbi[0,:] = np.log(Y_pred[0,start,:])

    for t in range(1, N):
        for label in range(L):
            update = np.add(viterbi[t-1,:], np.log(Y_pred[t,:-1,label]))
            backpointers[t, label] = np.argmax(update)
            viterbi[t, label] = update[backpointers[t, label]]

    # initialize as the last label
    label = np.argmax(viterbi[N-1,:])
    for t in reversed(range(N)):
        path.append(label)
        label = backpointers[t,label]
    path.reverse()
    return path

#################################################
# ================ DO NOT MODIFY ================
#################################################


def load_data(filename):
    """
        load data from filename and return a list of lists
        all_toks = [toks1, toks2, ...] where toks1 is
                a sequence of words (sentence)

        all_labs = [labs1, labs2, ...] where labs1 is a sequence of
                labels for the corresponding sentence
    """
    file = open(filename)
    all_toks = []
    all_labs = []
    toks = []
    labs = []
    for line in file:
        # Skip the license
        if "This data is licensed from" in line:
            continue
        cols = line.rstrip().split("\t")
        if len(cols) < 2:
            all_toks.append(toks)
            all_labs.append(labs)
            toks = []
            labs = []
            continue
        toks.append(cols[0])
        labs.append(cols[1])

    if len(toks) > 0:
        all_toks.append(toks)
        all_labs.append(labs)

    return all_toks, all_labs


def train(filename, data):
    """
        train a model to generate Y_pred
    """
    all_toks, all_labs = load_data(filename)
    vocab = {}

    # X_verbose is a list of feature objects for the entire train dataset.
    # Each feature object is a dictionary defined by
    # get_features and corresponds to a word
    # Y_verbose is a list of labels for all words in the entire train dataset
    X_verbose = []
    Y_verbose = []

    feature_counts=Counter()

    for i in range(len(all_toks)):
        toks = all_toks[i]
        labs = all_labs[i]
        for j in range(len(toks)):
            prev_lab = labs[j - 1] if j > 0 else "START"
            feats = get_features(j, toks, prev_lab, data)
            X_verbose.append(feats)
            Y_verbose.append(labs[j])

            for feat in feats:
                feature_counts[feat]+=1

    # construct label_vocab (dictionary) and feature_vocab (dictionary)
    # label_vocab[pos_tag_label] = index_for_the_pos_tag
    # feature_vocab[feature_name] = index_for_the_feature
    feature_id = 1
    label_id = 0

    # create unique integer ids for each label and each feature above the minimum count threshold
    for i in range(len(X_verbose)):
        feats = X_verbose[i]
        true_label = Y_verbose[i]

        for feat in feats:
            if feature_counts[feat] >= MINIMUM_FEAT_COUNT:
                if feat not in feature_vocab:
                    feature_vocab[feat] = feature_id
                    feature_id += 1
        if true_label not in label_vocab:
            label_vocab[true_label] = label_id
            label_id += 1

    # START has last id
    label_vocab["START"] = label_id

    # create train input and output to train the logistic regression model
    # create sparse input matrix

    # X is documents x features empty sparse matrix
    X = sparse.lil_matrix((len(X_verbose), feature_id))
    Y = []

    print_message("Number of features: %s" % len(feature_vocab))
    for i in range(len(X_verbose)):
        feats = X_verbose[i]
        true_label = Y_verbose[i]
        for feat in feats:
            if feat in feature_vocab:
                X[i, feature_vocab[feat]] = feats[feat]
        Y.append(label_vocab[true_label])

    # fit model
    log_reg = linear_model.LogisticRegression(C=L2_REGULARIZATION_STRENGTH, penalty='l2')
    log_reg.fit(sparse.coo_matrix(X), Y)

    return log_reg


def greedy_decode(Y_pred):
    """
        greedy decoding to get the sequence of label predictions
    """
    cur = label_vocab["START"]
    preds = []
    for i in range(len(Y_pred)):
        pred = np.argmax(Y_pred[i, cur])
        preds.append(pred)
        cur = pred
    return preds


def test(filename, log_reg, data):
    """
        predict labels using the trained model
        and evaluate the performance of model
    """
    all_toks, all_labs = load_data(filename)

    # possible output labels = all except START
    L = len(label_vocab) - 1

    correct = 0.
    total = 0.

    num_features = len(feature_vocab) + 1

    # for each sequence (sentence) in the test dataset
    for i in range(len(all_toks)):

        toks = all_toks[i]
        labs = all_labs[i]

        if len(toks) == 0:
            continue

        N = len(toks)

        X_test = []
        # N x prev_tag x cur_tag
        Y_pred = np.zeros((N, L + 1, L))

        # vector of true labels
        Y_test = []

        # for each token (word) in the sentence
        for j in range(len(toks)):

            true_label = labs[j]
            Y_test.append(true_label)

            # for each preceding tag of the word
            for possible_previous_tag in label_vocab:
                X = sparse.lil_matrix((1, num_features))

                feats = get_features(j, toks, possible_previous_tag, data)
                valid_feats = {}
                for feat in feats:
                    if feat in feature_vocab:
                        X[0, feature_vocab[feat]] = feats[feat]

                # update Y_pred with the probabilities of all current tags
                # given the current word, previous tag and other data/feature
                prob = log_reg.predict_proba(X)
                Y_pred[j, label_vocab[possible_previous_tag]] = prob

        # decode to get the predictions
        predictions = decode(Y_pred)

        # evaluate the performance of the model by checking predictions
        # against true labels
        for k in range(len(predictions)):
            if Y_test[k] in label_vocab and predictions[k] == label_vocab[Y_test[k]]:
                correct += 1
            total += 1

        print("Development Accuracy: %.3f (%s/%s)." % (correct / total, correct, total), end="\r")
    # print("Final Development Accuracy: %.3f (%s/%s)." % (correct / total, correct, total))


def print_message(m):
    num_stars = 10
    if verbose:
        print("*" * num_stars + m + "*" * num_stars)


def decode(Y_pred):
    """
        select the decoding algorithm
    """
    if use_greedy:
        return greedy_decode(Y_pred)
    else:
        return viterbi_decode(Y_pred)


# This function is for testing purpose. Please do not edit
def load_vars():
    with open('test.pickle', 'rb') as f:
        test_variables = pickle.load(f)

    return test_variables.label_vocab, test_variables.Y_pred


# usage: python memm_tagger.py -t wsj.pos.train wsj.pos.dev
def main():
    if sys.argv[1] == "-t":
        print_message("Initialize Data")
        data = initialize()
        print_message("Train Model")
        log_reg = train(sys.argv[2], data)
        print_message("Test Model")
        test(sys.argv[3], log_reg, data)
    # new flag for ablation testing
    # usage python memm_tagger.py -a 0 wsj.pos.train wsj.pos.dev
    # elif sys.argv[1] == "-a":
    #     global ABLATE
    #     ABLATE = int(sys.argv[2])
    #     print_message("Initialize Data")
    #     data = initialize()
    #     print_message("Train Model")
    #     log_reg = train(sys.argv[3], data)
    #     print_message("Test Model")
    #     test(sys.argv[4], log_reg, data)
    elif sys.argv[1] == "-v":
        label_vocab, Y_pred = load_vars()
        predicted_path = viterbi_decode(Y_pred)
        print(predicted_path)
    else:
        print("Usage: python memm_tagger.py -t wsj.pos.train wsj.pos.dev")


if __name__ == "__main__":
    main()
