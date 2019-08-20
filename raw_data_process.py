import os
import sys
import pickle
import string
from nltk.tokenize import word_tokenize

from config import *


table = str.maketrans('','',string.punctuation)

def isvalid_text_file(path):
    if not os.path.isfile(path):
        return False
    if not path.endswith('.txt'):
        return False
    return True


def build_dict(dict_path):
    assert isvalid_text_file(dict_path), 'Invalid dictionary file!'
    dicts = {}
    with open(dict_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            line = line.replace('\n', '')
            line = line.split(' ')
            dicts[str(line[0])] = int(line[1])

    return dicts


def build_stopwords(stw_path):
    assert isvalid_text_file(stw_path), 'Invalid stop words file!'
    stopwords = []
    with open(stw_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            stopwords.append(str(line))

    return stopwords


def filter_punctuations(tokens):
    global table
    stripped = [w.translate(table) for w in tokens]
    return stripped


def filter_non_alphabet(tokens):
    words = [word for word in tokens if word.isalpha()]
    return words


def filter_stopwords(tokens, stopwords):
    words = [word for word in tokens if not word in stopwords]
    return words


def clean_rawtext(sentences):
    results = []
    for sentence in sentences:
        if isinstance(sentence, str):
            tokens = word_tokenize(sentence)
        tokens = [token.lower() for torken in tokens]
        tokens = filter_punctuations(tokens)
        tokens = filter_non_alphabet(tokens)
        tokens = filter_stopwords(tokens)
        results.append(tokens)
    return results

def load_trainingset(nm_cmts_path, sr_cmts_path, stopwords):
    assert isvalid_text_file(nm_cmts_path), 'Invalid training normal comments!'
    assert isvalid_text_file(sr_cmts_path), 'Invalid training sara comments!'

    nm_cmts = []
    with open(nm_cmts_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n','')
            line_tokens = word_tokenize(line)


dicts = build_dict(DICT)
stopwords = build_stopwords(STWORDS)

