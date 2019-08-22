import os
import numpy as np
from config import *
from gensim.models import Word2Vec
import pickle

# model = Word2Vec.load('model.bin')
# model.load('model.bin')
# print(model.most_similar('muhamad'))
# print(MAX_LEN[-1])
max_len = []
with open('max_len.bin','rb') as f:
    max_len = pickle.load(f)

print(max_len)