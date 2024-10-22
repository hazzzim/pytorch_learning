from __future__ import unicode_literals, print_function, division

# importing all necessary modules
from gensim.models import Word2Vec
import gensim
from io import open


#import logging  # Setting up the loggings to monitor gensim
#logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)
#warnings.filterwarnings(action='ignore')

from io import open
import unicodedata
import re
import random

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# region preprocessing
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # start with EOS and SOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines....")

    # Read the file and split into 2 lines
    lines = open('../data/data_nlp_classification/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))


def word2vecEmbedding(pairs, epochs, hidden_size):
    input_sent = [inputlist[0].split(' ') for inputlist in pairs]
    input_emb_model = Word2Vec(min_count=1)
    input_emb_model.build_vocab(input_sent)  # prepare the model vocabulary
    input_emb_model.train(input_sent,vector_size=hidden_size, total_examples=input_emb_model.corpus_count, epochs=epochs)

    output_sent = [outputlist[1].split(' ') for outputlist in pairs]
    output_emb_model = Word2Vec(min_count=1)
    output_emb_model.build_vocab(output_sent)  # prepare the model vocabulary
    output_emb_model.train(output_sent,vector_size=hidden_size, total_examples=output_emb_model.corpus_count, epochs=epochs)

    return input_emb_model, output_emb_model


# To be deleted
# words_from_model = set(input_emb_model.wv.index_to_key)
# words_from_code = set(input_lang.index2word.values())
# not_in_model = words_from_code.difference(words_from_model)


inp_emb, out_emb = word2vecEmbedding(pairs, 1, 128)

print(1)
# endregion
