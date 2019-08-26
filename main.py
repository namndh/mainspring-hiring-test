import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import pickle
from torch.utils.data import DataLoader

from dataset import CommentDataset
from model.model import myModel
from config import *

def train():
    epochs = opts.epochs
    batch_size = opts.batch_size
    accumulate = opts.accumulate
    weights = opts.weights

    wdir = 'weights' + os.sep
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = myModel()
    model.double()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # scheduler = StepLR(optimizer, step_size=50, gamma=0.01)
    max_len = pickle.load('max_len.bin')
    max_len = itn(max_len[0])

    start_epoch = 0
    best_fitness = 0
    if opts.resume:
        if weights.endswith('.pt'):
            chkpt = torch.load(weights, map_location=device)
            model.load_state_dict(chkpt['model'])
            if chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']
                if chkpt.get('training_result') is not None:
                    print(chkpt['training_result'])
                start_epoch = chkpt['epoch'] + 1
                del chkpt

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opts.epoch * x) for x in [0.8,0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    train_set = CommentDataset(TRAIN_DATASET, 'word2vec.bin', max_len)
    val_set = CommentDataset(VAL_DATASET, 'word2vec.bin', max_len)
    test_set = CommentDataset(TEST_DATASET, 'word2vec.bin', max_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--data_folder', type=str, default='datasets', help='dir stores bin file of dataset')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--evaluate', action='store_true', help='evaluate trained model on test set')
    parser.add_argument('--weights', type=str, default='model/best.pt', help='dir store weight file for evaluating')
    opts = parser.parse_args()
    print(opts)




