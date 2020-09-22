#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

from typing import List, Set, Dict
import math
import numpy as np
import logging
from collections import Counter
# from IPython import embed

logger = logging.getLogger('language_modeling')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('lm.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


class Corpus(object):
    def __init__(self, path:str, add_bos:bool=True, add_eos:bool=True):
        logger.info('Parse and format corpus')
        self._corpus = []
        self._vocab = set()
        self._path = path
        self._num_sentences = 0
        self.parse_and_format_corpus(self._path, add_bos, add_eos)

    def parse_and_format_corpus(self, path:str, add_bos:bool=True, add_eos:bool=True):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    self._num_sentences += 1
                    l = line.strip().split()
                    l = [ w.lower() for w in l ]
                    for w in l:
                        self._vocab.add(w)
                    if add_eos:
                        l.append('</s>')
                    if add_bos:
                        l.insert(0,'<s>')
                    self._corpus.append(l)
        except IOError as e:
            logger.error(str(e))

    @property
    def corpus(self)->List[List[str]]:
        return(self._corpus)

    @property
    def vocab(self)->Set[str]:
        for sym in ['<s>', '</s>']:
            self._vocab.add(sym)
        return(self._vocab)

    @property
    def num_unique_words(self)->int:
        return(len(self.vocab))

    @property
    def num_sentences(self)->int:
        return(self._num_sentences)


class Unigram(object):
    def __init__(self, corpus:List[List[str]]):
        logger.info("read corpus")
        self._corpus = corpus
        self._unigram_counter = {}
        self._probs = {}
        self._tot_count = 0
        logger.info("count unigrams")
        self.count_unigrams()
    
    def count_unigrams(self):
        for sentence in self._corpus:
            for word in sentence:
                self._tot_count += 1
                if word in self._unigram_counter:
                    self._unigram_counter[word] += 1
                else:
                    self._unigram_counter[word] = 1
    
        for word in self._unigram_counter:
            self._probs[word] = self._unigram_counter[word] / self._tot_count

    def generate(self)->float:
        return(np.random.choice(list(self._probs.keys()),p=list(self._probs.values())))

    @property
    def counts(self)->Dict:
        return(self._unigram_counter)
    
    @property
    def probs(self)->Dict:
        return(self._probs)
    
    @property
    def tot_count(self)->int:
        return(self._tot_count)


class Bigram(Unigram):
    def __init__(self, corpus:List[List[str]]):
        Unigram.__init__(self, corpus)
        self._corpus = corpus
        self._bigram_counter = Counter()
        self.count_bigrams()
        self._probs = {}
        self.get_cond_probs()

    def count_bigrams(self):
        for line in self._corpus:
            self._bigram_counter.update([ (line[i], line[i+1]) for i in range(len(line)-1)])
    
    def get_cond_probs(self):
        # P(B|A) = C(A,B) / C(A)
        for k in self._bigram_counter:
            # count bigrams starting with k[0] in counter == count unigrams k[0]
            self._probs[k] = self._bigram_counter[k] / self._unigram_counter[k[0]]

    def generate(self):
        condition = '<s>'

    @property
    def counts(self)->Dict:
        return(dict(self._bigram_counter))
    
    @property
    def probs(self)->Dict:
        return(self._probs)