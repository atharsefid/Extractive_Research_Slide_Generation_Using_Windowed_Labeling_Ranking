import collections
import glob
import hashlib
import os
import pickle
import random
from random import shuffle

import numpy as np
import tensorflow as tf

import params

# Setting-up seeds
random.seed(2019)
np.random.seed(2019)
tf.set_random_seed(2019)


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())

    return lines


def hashhex(s):
    """
		Returns a heximal formatted SHA1 hash of the input string.
	"""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))

    return h.hexdigest()


def fix_missing_period(line):
    if "@highlight" in line:
        return line

    if line == "":
        return line

    if line[-1] in params.END_TOKENS:
        return line[: -1] + " " + line[-1]

    return line + " ."


def get_article_labels(sent_file):
    with open(sent_file, 'r') as sentFile:
        sents = sentFile.readlines()
    # with open(sent_file[:-len('.story.txt')] + '.windowed_summarunner_scores.txt', 'r') as labelFile:
    with open(sent_file[:-len('.story.txt')] + '.summarunner_scores.txt', 'r') as labelFile:
        labels = labelFile.readlines()
        labels = [int(label) for label in labels]
    return sents, labels


def article2ids(article_words, vocab):
    """
		This function converts given article words to ID's
	:param article_words: article tokens.
	:param vocab: The vocabulary object used for lookup tables, vocabulary etc.
	:return: The corresponding ID's and a list of OOV tokens.
	"""
    ids = []
    oovs = []
    unk_id = vocab.word2id(params.UNKNOWN_TOKEN)
    for word in article_words:
        i = vocab.word2id(word)
        if i == unk_id:  # Out of vocabulary words.
            if word in oovs:
                ids.append(vocab.size() + oovs.index(word))
            else:
                oovs.append(word)
                ids.append(vocab.size() + oovs.index(word))
        else:  # In vocabulary words.
            ids.append(i)

    return ids, oovs


def summary2ids(summary_words, vocab, article_oovs):
    """
	This function converts the given summary words to ID's
	:param summary_words: summary tokens.
	:param vocab: The vocabulary object used for lookup tables, vocabulary etc.
	:param article_oovs: OOV tokens in the input article.
	:return: The corresponding ID's.
	"""
    num_sent, sent_len = 0, 0
    if type(summary_words[0]) is list:
        num_sent = len(summary_words)
        sent_len = len(summary_words[0])

        cum_words = []
        for _, sent_words in enumerate(summary_words):
            cum_words += sent_words

        summary_words = cum_words

    ids = []
    unk_id = vocab.word2id(params.UNKNOWN_TOKEN)
    for word in summary_words:
        i = vocab.word2id(word)
        if i == unk_id:  # Out of vocabulary words.
            if word in article_oovs:  # In article OOV words.
                ids.append(vocab.size() + article_oovs.index(word))
            else:  # Both OOV and article OOV words.
                ids.append(unk_id)
        else:  # In vocabulary words.
            ids.append(i)

    if num_sent != 0:
        doc_ids = []
        for i in range(num_sent):
            doc_ids.append(ids[sent_len * i: sent_len * (i + 1)])

        ids = doc_ids

    return ids


class Vocab(object):
    def __init__(self, max_vocab_size, emb_dim=300, dataset_path='data/', glove_path='glove.6B/glove.6B.50d.txt',
                 vocab_path='data_files/vocab.txt', lookup_path='data_files/lookup.pkl'):

        self.max_size = max_vocab_size
        self._dim = emb_dim
        self.PathToGloveFile = glove_path
        self.PathToVocabFile = vocab_path
        self.PathToLookups = lookup_path

        # All train Stories.
        self._story_files = glob.glob(dataset_path + 'train/*.sents.txt')
        self.vocab = []  # Vocabulary

        # Create the vocab file.
        self.create_total_vocab()

        # Create the lookup tables.
        self.wvecs = []  # Word vectors.
        self._word_to_id = {}  # word to ID's lookups
        self._id_to_word = {}  # ID to word lookups

        self.create_lookup_tables()

        assert len(self._word_to_id.keys()) == len(self._id_to_word.keys()), "Both lookups should have same size."

    def size(self):
        return len(self.vocab)

    def word2id(self, word):
        """
      	This function returns the vocabulary ID for word if it is present. Otherwise, returns the ID
		for the unknown token.
		:param word: input word.
		:return: returns the ID.
		"""
        return self._word_to_id.get(word, self._word_to_id[params.UNKNOWN_TOKEN])

    def id2word(self, word_id):
        """
			This function returns the corresponding word for a given vocabulary ID.
		:param word_id: input ID.
		:return:  returns the word.
		"""
        if word_id in self._id_to_word:
            return self._id_to_word[word_id]
        else:
            raise ValueError("{} is not a valid ID.\n".format(word_id))

    def create_total_vocab(self):

        if os.path.isfile(self.PathToVocabFile):
            print("Vocab file exists! \n")

            vocab_f = open(self.PathToVocabFile, 'r')
            for line in vocab_f:
                word = line.split()[0]
                self.vocab.append(word)

            return
        else:
            print("Vocab file NOT found!! \n")
            print("Creating a vocab file! \n")

        vocab_counter = collections.Counter()

        for idx, story in enumerate(self._story_files):
            article, summary = get_article_labels(story)
            article = ' '.join(article)
            art_tokens = article.split(' ')  # Article tokens.
            tokens = art_tokens
            tokens = [t.strip() for t in tokens if t.strip() != '']
            # tokens = [t for t in tokens if t != '']  # Removing empty tokens.
            vocab_counter.update(tokens)  # Keeping a count of the tokens.
            print("\r{}/{} files read!".format(idx + 1, len(self._story_files)))

        print("\n Writing the vocab file! \n")
        f = open(self.PathToVocabFile, 'w')

        for word, count in vocab_counter.most_common(params.VOCAB_SIZE):
            f.write(word + ' ' + str(count) + '\n')
            self.vocab.append(word)

        f.close()

    def create_small_vocab(self):
        """
        This function selects a few words out of the total vocabulary.
        """

        # Read the vocab file and assign id's to each word till the max_size.
        vocab_f = open(self.PathToVocabFile, 'r')
        for line in vocab_f:
            word = line.split()[0]
            if word in [params.SENTENCE_START, params.SENTENCE_END, params.UNKNOWN_TOKEN,
                        params.PAD_TOKEN, params.START_DECODING, params.STOP_DECODING]:
                raise Exception(
                    '<s>, </s>, [UNK], [PAD], [START], [STOP] shouldn\'t be in the vocab file, but %s is' % word)

            self.vocab.append(word)
            print("\r{}/{} words created!".format(len(self.vocab), self.max_size))

            if len(self.vocab) == self.max_size:
                print("\n Max size of the vocabulary reached! Stopping reading! \n")
                break

    def create_lookup_tables(self):
        """
        This function creates lookup tables for word vectors, word to IDs
        and ID to words. First max_size words from GloVe that are also found in the small vocab are used
        to create the lookup tables.
        """

        if os.path.isfile(self.PathToLookups):
            print('\n Lookup tables found :) \n')
            f = open(self.PathToLookups, 'rb')
            data = pickle.load(f)

            self._word_to_id = data['word2id']
            self._id_to_word = data['id2word']
            self.wvecs = data['wvecs']
            self.vocab = list(self._word_to_id.keys())

            print('Lookup tables collected for {} tokens.\n'.format(len(self.vocab)))
            return
        else:
            print('\n Lookup files NOT found!! \n')
            print('\n Creating the lookup tables! \n')

        self.create_small_vocab()  # Creating a small vocabulary.
        self.wvecs = []  # Word vectors.

        glove_f = open(self.PathToGloveFile, 'r', encoding='utf8')
        count = 0

        # [UNK], [PAD], [START] and [STOP] get ids 0, 1, 2, 3.
        for w in [params.UNKNOWN_TOKEN, params.PAD_TOKEN, params.START_DECODING, params.STOP_DECODING]:
            self._word_to_id[w] = count
            self._id_to_word[count] = w
            self.wvecs.append(np.random.uniform(-0.1, 0.1, (self._dim,)).astype(np.float32))
            count += 1

            print("\r Created tables for {}".format(w))

        for line in glove_f:
            vals = line.rstrip().split(' ')
            w = vals[0]
            vec = np.array(vals[1:]).astype(np.float32)

            if w in self.vocab:
                self._word_to_id[w] = count
                self._id_to_word[count] = w
                self.wvecs.append(vec)
                count += 1

                print("\r Created tables for' {}".format(w))

            if count == self.max_size:
                print("\r Maximum vocab size reached! \n")
                break

        print("\n Lookup tables created for {} tokens. \n".format(count))

        self.wvecs = np.array(self.wvecs).astype(np.float32)  # Converting to a Numpy array.
        self.vocab = list(self._word_to_id.keys())  # Adjusting the vocabulary to found pre-trained vectors.

        # Saving the lookup tables.
        f = open(self.PathToLookups, 'wb')
        data = {'word2id': self._word_to_id,
                'id2word': self._id_to_word,
                'wvecs': self.wvecs}
        pickle.dump(data, f)


class DataGenerator(object):

    def __init__(self, path_to_dataset, max_inp_seq_len, max_out_seq_len, vocab, use_pgen=False, use_sample=False):
        # Train files.

        self.train_files = glob.glob('data/train/*.sents.txt')
        self.num_train_examples = len(self.train_files)
        shuffle(self.train_files)

        # Validation files.
        self.val_files = glob.glob('data/val/*.sents.txt')
        self.num_val_examples = len(self.val_files)
        shuffle(self.val_files)

        # Test files.
        self.test_files = glob.glob('data/test/*.sents.txt')
        self.num_test_examples = len(self.test_files)
        # shuffle(self.test_files)

        self._max_enc_steps = max_inp_seq_len  # Max. no. of tokens in the input sequence.

        self.vocab = vocab  # Vocabulary instance.

        self.ptr = 0  # Pointer for batching the data.

        if use_sample:
            # **************************** PATCH ************************* #
            self.train_files = self.train_files[:3]
            self.num_train_examples = len(self.train_files)
            self.val_files = self.val_files[:3]
            self.num_val_examples = len(self.val_files)
            self.test_files = self.test_files[:3]
            self.num_test_examples = len(self.test_files)
        # **************************** PATCH ************************* #

        print("Split the data as follows:\n")
        print("\t\t Training: {} examples. \n".format(self.num_train_examples))
        print("\t\t Validation: {} examples. \n".format(self.num_val_examples))
        print("\t\t Test: {} examples. \n".format(self.num_test_examples))

    def get_test_files(self):
        return self.test_files

    def get_train_val_batch(self, split='train'):
        if split == 'train':
            num_examples = self.num_train_examples
            files = self.train_files
        elif split == 'val':
            num_examples = self.num_val_examples
            files = self.val_files
        else:
            raise ValueError("split is neither train nor val. check the function call!")

        # Shuffle files at the start of an epoch.
        if self.ptr == 0:
            shuffle(files)
        enc_inp = []
        dec_inp = []
        # dec_out = []

        for point in range(params.batch_size):
            if self.ptr >= num_examples:
                break
            article, labels = get_article_labels(files[self.ptr])

            if len(article) > params.doc_size:
                article = article[:params.doc_size]
            elif len(article) < params.doc_size:
                article = article + [''] * (params.doc_size - len(article))

            enc_inputs = []
            for i, sent_inp_tokens in enumerate(article):
                sent_inp_tokens = sent_inp_tokens.split()
                # Article Tokens
                if len(sent_inp_tokens) >= self._max_enc_steps:  # Truncate.
                    sent_inp_tokens = sent_inp_tokens[: self._max_enc_steps]
                else:  # Pad.
                    sent_inp_tokens += (self._max_enc_steps - len(sent_inp_tokens)) * [params.PAD_TOKEN]

                # sentence token ids
                sent_inp_ids = [self.vocab.word2id(w) for w in sent_inp_tokens]  # Word to ID's
                enc_inputs.append(sent_inp_ids)

            # doc size
            if len(labels) > params.doc_size:  # Truncate.
                labels = labels[: params.doc_size]

            # Decoder Input
            dec_inp_labels = labels
            if len(dec_inp_labels) < params.doc_size:
                dec_inp_labels += (params.doc_size - len(dec_inp_labels)) * [params.PAD_Label]

            if len(enc_inputs) != len(labels):
                raise ValueError('The # of sentences and labels are not equal.')

            enc_inp.append(enc_inputs)
            dec_inp.append(labels)
            # dec_out.append(labels)

            self.ptr += 1

        enc_inp_np = np.array(enc_inp).astype(np.int32)
        dec_inp_np = np.array(dec_inp).astype(np.int32)
        # dec_out_np = np.array(dec_out).astype(np.int32)

        if enc_inp_np.shape[0] != params.batch_size:
            enc_inp_np = np.concatenate([enc_inp_np, np.zeros(
                [params.batch_size - enc_inp_np.shape[0], params.doc_size, enc_inp_np.shape[-1]])], axis=0)
            dec_inp_np = np.concatenate(
                [dec_inp_np, np.zeros([params.batch_size - dec_inp_np.shape[0], params.doc_size])], axis=0)
        # dec_out_np = np.concatenate([dec_out_np, np.zeros([params.batch_size-dec_inp_np.shape[0],params.doc_size])], axis=0)

        # Resetting the pointer after the last batch
        if self.ptr == num_examples:
            self.ptr = 0
        batch = [enc_inp_np, dec_inp_np]
        return batch

    def get_test_batch(self):
        num_examples = self.num_test_examples
        files = self.test_files[:]

        enc_inp = []
        dec_inp = []
        filenames = []
        for point in range(params.batch_size):
            if self.ptr >= num_examples:
                break
            article, labels = get_article_labels(files[self.ptr])
            filenames.append(files[self.ptr])

            if len(article) > params.doc_size:
                article = article[:params.doc_size]
            elif len(article) < params.doc_size:
                article = article + [''] * (params.doc_size - len(article))

            enc_inputs = []
            for i, sent_inp_tokens in enumerate(article):
                sent_inp_tokens = sent_inp_tokens.split()
                # Article Tokens
                if len(sent_inp_tokens) >= self._max_enc_steps:  # Truncate.
                    sent_inp_tokens = sent_inp_tokens[: self._max_enc_steps]
                else:  # Pad.
                    sent_inp_tokens += (self._max_enc_steps - len(sent_inp_tokens)) * [params.PAD_TOKEN]

                # sentence token ids
                sent_inp_ids = [self.vocab.word2id(w) for w in sent_inp_tokens]  # Word to ID's
                enc_inputs.append(sent_inp_ids)

            # doc size
            if len(labels) > params.doc_size:  # Truncate.
                labels = labels[: params.doc_size]

            # Decoder Input
            dec_inp_labels = labels
            if len(dec_inp_labels) < params.doc_size:
                dec_inp_labels += (params.doc_size - len(dec_inp_labels)) * [params.PAD_Label]

            if len(enc_inputs) != len(labels):
                raise ValueError('The # of sentences and labels are not equal.')

            enc_inp.append(enc_inputs)
            dec_inp.append(labels)

            self.ptr += 1

        enc_inp_np = np.array(enc_inp).astype(np.int32)
        dec_inp_np = np.array(dec_inp).astype(np.int32)

        if enc_inp_np.shape[0] != params.batch_size:
            enc_inp_np = np.concatenate([enc_inp_np, np.zeros(
                [params.batch_size - enc_inp_np.shape[0], params.doc_size, enc_inp_np.shape[-1]])], axis=0)
            dec_inp_np = np.concatenate(
                [dec_inp_np, np.zeros([params.batch_size - dec_inp_np.shape[0], params.doc_size])], axis=0)

        # Resetting the pointer after the last batch
        if self.ptr == num_examples:
            self.ptr = 0
        batch = [enc_inp_np, dec_inp_np]
        return batch, filenames

    def get_batch(self, split='train'):
        if split == 'train' or split == 'val':
            return self.get_train_val_batch(split)
        elif split == 'test':
            return self.get_test_batch()
        else:
            raise ValueError('split should be either of train/val/test only!! \n')
