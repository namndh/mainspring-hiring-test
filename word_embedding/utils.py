import os
from config import *
from raw_data_process import clean_rawtext

def load_sentences(path, stopwords):
    sentences = []
    assert os.path.isfile(path), 'File does not exist!'
    assert path.endswith('.txt'), 'Invalid file type!'
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n','')
            tokens = clean_rawtext(line,stopwords)
            sentences.append(tokens)
    return sentences



def load_all_sentences(stopwords):

    all_sentences = load_sentences(TRAINING_NORMAL_CMT,stopwords) + load_sentences(TRAINING_SARA_CMT,stopwords) + load_sentences(TEST_NORMAL_CMT, stopwords) \
                    + load_sentences(TEST_SARA_CMT, stopwords)
    return all_sentences







