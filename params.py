# -*- coding: utf-8 -*-
dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'

# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]

# Vocabulary id's for sentence start, end, pad, unknown token, start and stop decoding.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = 3
STOP_DECODING = 4
PAD_TOKEN = 2
VOCAB_SIZE = 200000

PAD_Label = 0
doc_size = 500
max_vocab_size = 200000
max_inp_seq_len = 50
max_out_seq_len = 50
batch_size = 16
