from .utils import *
from config import *
from raw_data_process import *
import multiprocessing
from gensim.models import Word2Vec
import logging
import pickle

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)

stopwords = build_stopwords(STWORDS)
all_sentences = load_all_sentences(stopwords)

max_len = max([len(sentence) for sentence in all_sentences])

with open('max_len.bin', 'wb+') as f:
    pickle.dump([max_len,], f)

cores = multiprocessing.cpu_count()

w2v_model  = Word2Vec(min_count=10,window=2,size=300,sample=6e-5,alpha=0.01, min_alpha=0.0005,workers=cores-2)

w2v_model.build_vocab(sentences=all_sentences, progress_per=10000)

w2v_model.train(all_sentences, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1)

w2v_model.save('word2vec.bin')
