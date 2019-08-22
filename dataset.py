import os 
import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import pickle


def padding_text(max_len, sentence):
    padding_vec = np.zeros((1,sentence.shape[1]))
    for i in range(abs(max_len-sentence.shape[0])):
        sentence  = np.append(sentence, padding_vec, axis=0)
    return sentence


class CommentDataset(Dataset):
    def __init__(self, bin_file, root_dir, word2vec_model_file, max_len,  transform=None):
        assert os.path.isfile(bin_file), 'Dataset is not existed!'
        assert bin_file.endswith('.bin'), 'Invalid file format!'
        assert os.path.isfile(word2vec_model_file), 'Word2vec model is '
        with open(bin_file, 'rb') as f:
            self.data = pickle.load(f)
        self.root_dir = root_dir
        self.transform = transform
        self.word2vec = Word2Vec.load(word2vec_model_file)
        self.max_len = max_len
        # if data.shape[]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item[0]
        sentence = item[1]
        sentence_vector = np.asarray()
        for word in sentence

sentence = np.random.rand(120,300)

sentence = padding_text(130,sentence)
print(sentence.shape)
# print(sentence)


