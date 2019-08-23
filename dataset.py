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
    for i in range(abs(max_len-int(sentence.shape[0]))):
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
        print(sentence)
        print(label)
        sentence_vector = []
        for word in sentence:
            word_vector = self.word2vec[word]
            # print(word_vector)
            sentence_vector.append(word_vector)
        print(len(sentence_vector))
        sentence_vector = np.asarray(sentence_vector)
        sentence_vector = np.stack(sentence_vector)
        print(sentence_vector.shape)
        sentence_vector = padding_text(self.max_len, sentence_vector)
        print(sentence_vector.shape)
        if sentence_vector.shape[0] > self.max_len:
            exit(10)
        sentence_vector = torch.from_numpy(sentence_vector)
        sentence_vector = sentence_vector.unsqueeze(0)

        print(sentence_vector.shape)
        if self.transform:
            sentence_vector = self.transform(sentence_vector)
        print(sentence_vector)
        return (sentence_vector, label)

# with open('max_len.bin', 'rb') as f:
#     max_len = pickle.load(f)
#
# # #
# # model = Word2Vec.load('model.bin')
# #
# # print(model['kaya'])
# # data_transform = transforms.Compose([transforms.ToTensor()])
#
# train_dataset = CommentDataset('datasets/training_dataset.bin', '', 'model.bin', int(max_len[0]))
# print(len(train_dataset))
# # print(sentence)
# sample = train_dataset[10]
#
#
# print(sample[0].shape)
# net = myModel().to('cpu')
# net.double()
# print(net)
# # print(net.dropout)
# net.forward(sample[0])
# # print(net)