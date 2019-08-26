import os 
import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import pickle
from model.model import myModel
from torchvision import transforms


def padding_text(max_len, sentence):
    padding_vec = np.zeros((1,sentence.shape[1]))
    for i in range(abs(int(max_len)-int(sentence.shape[0]))):
        sentence  = np.append(sentence, padding_vec, axis=0)
    return sentence


class CommentDataset(Dataset):
    def __init__(self, bin_file, word2vec_model_file, max_len_file, transform=None):
        assert os.path.isfile(bin_file), 'Dataset is not existed!'
        assert bin_file.endswith('.bin'), 'Invalid file format!'
        assert os.path.isfile(word2vec_model_file), 'Word2vec model is not existed!'
        with open(bin_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.transform = transform
        self.word2vec = Word2Vec.load(word2vec_model_file)
        with open(max_len_file, 'rb') as f:
            self.max_len = int(pickle.load(f)[0])
        # if data.shape[]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item[0]
        sentence = item[1]
        sentence_vector = []
        if not len(sentence) > 0:
            exit(100)
        for word in sentence:
            try:
                word_vector = self.word2vec[word]
                sentence_vector.append(word_vector)
            except Exception:
                print('{}-{}'.format(sentence,word))
                exit(100)
        
        sentence_vector = np.asarray(sentence_vector)
        sentence_vector = np.stack(sentence_vector)
        
        sentence_vector = padding_text(self.max_len, sentence_vector)
        
        if sentence_vector.shape[0] > self.max_len:
            exit(100)
        sentence_vector = torch.from_numpy(sentence_vector)
        sentence_vector.float()
        sentence_vector = sentence_vector.unsqueeze(0)

        
        if self.transform:
            sentence_vector = self.transform(sentence_vector)

        return (sentence_vector, label)
