import os
import sys
import pickle

from config import *
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

dicts = build_dict(DICT)
stopwords = build_stopwords(STWORDS)

