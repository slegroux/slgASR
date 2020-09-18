#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

import pandas as pd
from IPython import embed
from typing import List, Set
import math
import numpy as np
import logging

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
    def __init__(self, path:str, add_bos=True, add_eos=True):
        self._corpus = []
        self._vocab = set()
        self._path = path
        self._num_sentences = 0
        self.parse_and_format_corpus(self._path, add_bos, add_eos)

    def parse_and_format_corpus(self, path:str, add_bos=True, add_eos=True):
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


class LanguageModel(object):
    def __init__(self, corpus:Corpus, order=1):
        self._corpus = corpus
        self._unigrams = {}
        self._unigram_counter = {}
        self._bigrams = {}
        self._bigram_counter = {}
        self._trigrams = {}
        if order == 1:
            self.compute_unigrams()
        elif order == 2:
            self.compute_unigrams()
            self.compute_bigrams()
    
    def compute_unigrams(self):
        words_count = 0
        for sentence in self._corpus:
            for word in sentence:
                words_count += 1
                if word in self._unigrams.keys():
                    self._unigrams[word] += 1
                else:
                    self._unigrams[word] = 1
        for k,v in self._unigrams.items():
            self._unigrams[k] = math.log10(v/words_count)
    
    def compute_bigrams(self):
        if not self._unigrams:
            self.compute_unigrams()
        bigrams_count = 0 # tot nber of bigrams
        for sentence in self._corpus:
            for i in range(len(sentence)-1):
                bg = (sentence[i], sentence[i+1])
                print(bg)
                bigrams_count += 1
                if bg in self._bigram_counter.keys():
                    self._bigram_counter[bg] += 1
                else:
                    self._bigram_counter[bg] = 1

        for k,v in self._bigram_counter.items():
            # P(b|a) = P(a&b)/P(a)
            self._bigrams[k] = math.log10(self._bigram_counter[k] / bigrams_count) - self._unigrams[k[0]]
    
    @property
    def vocabulary(self):
        return(set(self._unigrams.keys()))


class Unigram(object):
    def __init__(self, corpus:List[List[str]]):
        logger.info("read corpus")
        self._corpus = corpus
        logger.info("count unigrams")
        self._counts = {}
        self._probs = {}
        self._tot_count = 0
        self.count_unigrams()
    
    def count_unigrams(self):
        for sentence in self._corpus:
            for word in sentence:
                self._tot_count += 1
                if word in self._counts:
                    self._counts[word] += 1
                else:
                    self._counts[word] = 1
    
        for word in self._counts:
            self._probs[word] = self._counts[word] / self._tot_count

    def generate(self)->float:
        return(np.random.choice(list(self._probs.keys()),p=list(self._probs.values())))

    @property
    def counts(self):
        return(self._counts)
    
    @property
    def probs(self):
        return(self._probs)
    
    @property
    def tot_count(self):
        return(self._tot_count)

