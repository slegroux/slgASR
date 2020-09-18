#!/usr/bin/env python
# to run tests:
# pytest -p no:warnings -svDATA_FOLDER +'.py

import pytest
from lm import Corpus, Unigram
from IPython import embed
import os

DATA_FOLDER='data/tests'

# fixture to init global variables
@pytest.fixture(scope="module")
def data():
    data_ = {
        'corpus': DATA_FOLDER +'/lm/corpus.txt',
        'large_corpus': DATA_FOLDER +'/lm/brown_100.txt'
        }
    return data_

# test WavFile class
def test_corpus(data):
    c = Corpus(data['corpus'])
    assert c.corpus == [
        ['<s>', 'i', 'am', 'sam', '</s>'],
        ['<s>', 'sam', 'i', 'am', '</s>'],
        ['<s>', 'i', 'do', 'not', 'like', 'green', 'eggs', 'and', 'ham', '</s>']
    ]

def test_vocab(data):
    c = Corpus(data['corpus'])
    assert c.vocab == {'eggs', 'sam', 'do', '</s>', 'like', 'not', 'green', 'ham', '<s>', 'and', 'i', 'am'}

def test_counting(data):
    c = Corpus(data['corpus'])
    assert c.num_unique_words == 12
    assert c.num_sentences == 3

def test_unigram_counts(data):
    c = Corpus(data['corpus'])
    u = Unigram(c.corpus)
    assert u.counts == {'<s>': 3, 'i': 3, 'am': 2, 'sam': 2, '</s>': 3, 'do': 1, 'not': 1, 'like': 1, \
         'green': 1, 'eggs': 1, 'and': 1, 'ham': 1}
    assert u.probs == {'<s>': 0.15, 'i': 0.15, 'am': 0.1, 'sam': 0.1, '</s>': 0.15, 'do': 0.05, 'not': 0.05, \
         'like': 0.05, 'green': 0.05, 'eggs': 0.05, 'and': 0.05, 'ham': 0.05}
    assert u.tot_count == 20

def test_unigram_generate(data):
    c = Corpus(data['corpus'])
    u = Unigram(c.corpus)
    seq = []
    for i in range(10):
        seq.append(u.generate())
    print(seq)

def test_large_corpus(data):
    # brown corpus already has eos/beos added to text file
    c = Corpus(data['large_corpus'], add_eos=False, add_bos=False)
    u = Unigram(c.corpus)
    assert u.probs['all'] == 0.0004051863857374392
    assert u.probs['resolution'] == 0.003646677471636953

def test_unigram(data):
    pass