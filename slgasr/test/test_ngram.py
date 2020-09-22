#!/usr/bin/env python
# to run tests:
# pytest -p no:warnings -svDATA_FOLDER +'.py

import pytest
from slgasr.ngram import Corpus, Unigram, Bigram
from IPython import embed
import os
from pathlib import Path

DATA_FOLDER= str(Path(__file__).parent.parent / "data/tests")

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

    c = Corpus(data['corpus'], add_bos=False, add_eos=False)
    assert c.corpus == [
        ['i', 'am', 'sam'],
        ['sam', 'i', 'am'],
        ['i', 'do', 'not', 'like', 'green', 'eggs', 'and', 'ham']
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

def test_bigram_count(data):
    c = Corpus(data['corpus'])
    b = Bigram(c.corpus)
    
    assert b.counts == {('<s>', 'i'): 2, ('i', 'am'): 2, ('am', 'sam'): 1, ('sam', '</s>'): 1, \
        ('<s>', 'sam'): 1, ('sam', 'i'): 1, ('am', '</s>'): 1, ('i', 'do'): 1, ('do', 'not'): 1, \
        ('not', 'like'): 1, ('like', 'green'): 1, ('green', 'eggs'): 1, ('eggs', 'and'): 1, \
        ('and', 'ham'): 1, ('ham', '</s>'): 1}
    
    assert b.probs == {('<s>', 'i'): 0.6666666666666666, ('i', 'am'): 0.6666666666666666, \
        ('am', 'sam'): 0.5, ('sam', '</s>'): 0.5, ('<s>', 'sam'): 0.3333333333333333, \
        ('sam', 'i'): 0.5, ('am', '</s>'): 0.5, ('i', 'do'): 0.3333333333333333, \
        ('do', 'not'): 1.0, ('not', 'like'): 1.0, ('like', 'green'): 1.0, \
        ('green', 'eggs'): 1.0, ('eggs', 'and'): 1.0, ('and', 'ham'): 1.0, ('ham', '</s>'): 1.0}

def test_large_corpus(data):
    # brown corpus already has eos/beos added to text file
    c = Corpus(data['large_corpus'], add_eos=False, add_bos=False)
    u = Unigram(c.corpus)
    b = Bigram(c.corpus)
    assert u.probs['all'] == 0.0004051863857374392
    assert u.probs['resolution'] == 0.003646677471636953
    # p(the|all)
    assert b.probs[('all', 'the')] == 1.0
    # p(jury|the)
    assert b.probs[('the', 'jury')] == 0.08333333333333333
