import os
import sys
import pickle
import string
from nltk.tokenize import word_tokenize
from random import shuffle

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


def filter_indo_stopwords(tokens, stopwords):
    words = [word for word in tokens if not word in stopwords]
    return words

def filter_eng_stopwords(tokens, path):
    stopwords = build_stopwords(path)
    words = [word for word in tokens if not word in stopwords]
    return words


def clean_rawtext(sentence, stopwords):
    if isinstance(sentence, str):
        tokens = word_tokenize(sentence)
    if isinstance(sentence, list):
        tokens = sentence
    else:
        assert  True, "Invalid type of sentence!"
    tokens = [token.lower() for token in tokens]
    tokens = filter_punctuations(tokens)
    tokens = filter_non_alphabet(tokens)
    tokens = filter_indo_stopwords(tokens, stopwords)
    tokens = filter_eng_stopwords(tokens, 'stopwords-en.txt')
    return tokens


def load_dataset(nm_cmts_path, sr_cmts_path, stopwords):
    assert isvalid_text_file(nm_cmts_path), 'Invalid normal comments!'
    assert isvalid_text_file(sr_cmts_path), 'Invalid sara comments!'

    nm_cmts = []
    with open(nm_cmts_path, 'r') as f:
        lines = f.readlines()
        print(len(lines))
        for line in lines:
            line = line.replace('\n','')
            tokens = clean_rawtext(line, stopwords)
            nm_cmts.append([0,tokens])
    print(len(nm_cmts))
    sr_cmts = []
    with open(sr_cmts_path, 'r') as f:
        lines = f.readlines()
        print(len(lines))
        for line in lines:
            line = line.replace('\n','')
            tokens = clean_rawtext(line, stopwords)
            sr_cmts.append([0,tokens])
    print(len(sr_cmts))
    dataset = nm_cmts + sr_cmts
    shuffle(dataset)
    return dataset

dicts = build_dict(DICT)
stopwords = build_stopwords(STWORDS)
training_set = load_dataset(TRAINING_NORMAL_CMT,TRAINING_SARA_CMT,stopwords)
shuffle(training_set)
pivot = int(len(training_set)*0.8)

train_set = training_set[:pivot]
val_set = training_set[pivot:]

with open(TRAIN_DATASET, 'wb+') as f:
    pickle.dump(train_set, f)
with open(VAL_DATASET, 'wb+') as f:
    pickle.dump(val_set, f)

testing_set = load_dataset(TEST_NORMAL_CMT,TEST_SARA_CMT, stopwords)
with open(TEST_DATASET, 'wb+') as f:
    pickle.dump(testing_set,f)
