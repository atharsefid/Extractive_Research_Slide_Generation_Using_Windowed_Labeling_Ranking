import json
import string
from glob import glob

import nltk
import numpy as np
import rouge
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer


class sent_ranker:

    def __init__(self):
        self.stopset = set(stopwords.words('english')).union(string.punctuation)
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.snowball_stemmer = SnowballStemmer('english')
        self.rouge = rouge.Rouge()
        self.covered_sents = 0
        self.needed_sents = 0
        self.evaluator = rouge.Rouge(metrics=['rouge-n'],
                                     max_n=2,
                                     limit_length=False,
                                     apply_avg=True,
                                     apply_best=None,
                                     alpha=0.5,  # Default F1_score
                                     weight_factor=1.2,
                                     stemming=True)
        self.total_score = 0

    def normalize_tokens(self, text):
        tokens = word_tokenize(text)
        tokens = [self.snowball_stemmer.stem(token.lower()) for token in tokens if token.lower() not in self.stopset]
        return tokens

    def normalize(self, text):
        tokens = word_tokenize(text)
        tokens = [self.wordnet_lemmatizer.lemmatize(token.lower(), pos='n') for token in tokens if
                  token.lower() not in self.stopset]
        return ' '.join(tokens)

    def cosine_similarity(self, a, b):
        a = a.toarray()[0]
        b = b.toarray()[0]
        return np.dot(a, b) / (self.l2_norm(a) * self.l2_norm(b))

    def jaccard_similarity(self, tokens1, tokens2):
        if len(tokens1.union(tokens2)) == 0:
            return 0
        return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

    def ngram(self, t1):
        gram123 = set()
        gram123 = gram123.union(t1)
        bi_gram = set(nltk.bigrams(t1))
        gram123 = gram123.union(bi_gram)
        three_gram = set(nltk.ngrams(t1, 3))
        gram123 = gram123.union(three_gram)
        return gram123

    def read_ppt_txt(self, pptfile):
        ppttxt = open(pptfile, 'r', encoding='utf-8', errors='ignore')
        slides = []
        slide = ''
        for line in ppttxt:
            if line.startswith('\f'):
                slides.append(slide)
                slide = line[1:]
            else:
                slide = slide + line
        return slides

    def summarunner_rank(self, input):
        """
        This function ranks sents based on summarunner paper.
		It considers one sentence at a time and assignes 1 to the score if it increases the rouge score.
        """
        textfile, pptfile = input[0], input[1]
        pdf_file = open(textfile, 'r', encoding='utf-8', errors='ignore')
        article = json.load(pdf_file)
        name_parts = textfile.split('/')
        filename = name_parts[-3]
        print(filename)
        path = '/'.join(name_parts[:-1])
        outfile = open(path + '/' + filename + '.summarunner_scores.txt', 'w')
        sentences = [''.join([token['originalText'] + token['after'] for token in sentence['tokens']]) for sentence
                     in article['sentences']]

        max_size = 0.2 * len(' '.join(sentences))
        slides = open(pptfile, 'r').readlines()
        normalized_slides = [self.normalize(slide) for slide in slides]
        normalized_sentences = [self.normalize(sentence) for sentence in sentences]

        # Joining all article sentences together into a string.
        summary = ' '.join(normalized_slides)

        labels = [0] * len(normalized_sentences)

        cur_summary = ''
        prev_score = 0
        needed_sents = 0
        covered_sents = 0
        flag = False
        for i, sent in enumerate(normalized_sentences):
            cur_summary += sent
            scores = self.evaluator.get_scores(cur_summary, [summary])
            fscore = scores['rouge-1']['r']
            if fscore > prev_score:
                # labels[i] = fscore - prev_score  # add sentence i to the summary
                labels[i] = 1
                needed_sents += len(sent)
                if not flag:
                    covered_sents += len(sent)
                prev_score = fscore
            else:
                cur_summary = cur_summary[:-len(sent)]
            if flag is False and len(cur_summary) > max_size:
                flag = i
            if flag:
                labels[i] = 0
            outfile.write(str(labels[i]) + '\n')
        self.covered_sents += covered_sents
        self.needed_sents += needed_sents
        outfile.close()

    def windowed_summarunner_rank(self, input):
        textfile, pptfile, window = input[0], input[1], input[2]
        pdf_file = open(textfile, 'r', encoding='utf-8', errors='ignore')
        article = json.load(pdf_file)
        name_parts = textfile.split('/')
        filename = name_parts[-3]
        print(filename)
        path = '/'.join(name_parts[:-1])
        outfile = open(path + '/' + filename + '.windowed_summarunner_scores_{}.txt'.format(window), 'w')
        sentsfile = open(path + '/' + filename + '.sents.txt', 'w')
        sentences = [''.join([token['originalText'] + token['after'] for token in sentence['tokens']]) for sentence in
                     article['sentences']]

        max_size = 0.2 * len(' '.join(sentences))
        slides = open(pptfile, 'r').readlines()
        normalized_slides = [self.normalize(slide) for slide in slides]
        normalized_sentences = [self.normalize(sentence) for sentence in sentences]
        # Joining all article sentences together into a string.
        summary = ' '.join(normalized_slides)
        cur_summary = ''
        covered_sents = set()
        useless_sents = set()
        i = 0
        max_score = -1
        prev_score = 0
        while i < len(normalized_sentences) and len(covered_sents) + len(useless_sents) < len(normalized_sentences):
            max_index = -1
            max_score = -1

            for w in range(window):
                if i + w < len(normalized_sentences) and i + w not in covered_sents and i + w not in useless_sents:
                    scores = self.evaluator.get_scores(cur_summary + ' ' + normalized_sentences[i + w], [summary])
                    if scores['rouge-1']['r'] > max_score and scores['rouge-1']['r'] > prev_score:
                        max_score = scores['rouge-1']['r']
                        max_index = i + w
                    elif scores['rouge-1']['r'] <= prev_score:
                        useless_sents.add(i + w)

            if max_index != -1:
                cur_summary += normalized_sentences[max_index]
                prev_score = max_score
                covered_sents.add(max_index)
            if len(cur_summary) > max_size:
                break
            i += w + 1
            if i > len(normalized_sentences):
                i = 0
        if max_score != -1:
            self.total_score += max_score
        for sent in sentences:
            sentsfile.write(' '.join(sent.split()).strip() + '\n')

        for c in range(len(normalized_sentences)):
            if c in covered_sents:
                outfile.write('1\n')
            else:
                outfile.write('0\n')
        outfile.close()


from multiprocess import Pool

sr = sent_ranker()
window = 3
inputs = []
for i in range(4982):
    sentence_path = glob('data/' + str(i) + '/grobid/sections.stanfordnlp.json')[0]
    ppt_path = glob('data/' + str(i) + '/clean_tika.txt')[0]
    inputs.append((sentence_path, ppt_path, window))
pool = Pool(40)
pool.map(sr.windowed_summarunner_rank, inputs)
# pool.map(sr.summarunner_rank, inputs)
pool.close()
pool.join()
