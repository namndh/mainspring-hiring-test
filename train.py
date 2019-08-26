import os
import sys
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import myModel
from dataset import CommentDataset
from config import *


def train(opts):
    epochs = opts.epochs
    batch_size = opts.batch_size

    wdir = os.path.join(ROOT_DIR, 'weights')
    best_weight = os.path.join(wdir, 'best.pt')
    last_weight = os.path.join(wdir, 'last.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = myModel()
    # model.double()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_fitness = 0

    if opts.resume:
        assert os.path.isdir(wdir), 'weight folder does not exist!'
        assert os.path.isfile(last_weight), 'the laster weight file does not exist!'
        try:
            chkpt = torch.load(last_weight, map_location=device)
            model.load_state_dict(chkpt['model'])

            if chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])
            
            if chkpt.get('training_results') is not None:
                print(chkpt['training_results'])
            
            start_epoch = chkpt['epoch'] + 1
            del chkpt
        except Exception:
            SystemExit('Error: Can not load pretrained model!')
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opts.epochs*x) for x in [0.8,0.9]], gamma=0.2)
    scheduler.last_epoch = start_epoch-1

    train_dataset = CommentDataset(TRAIN_DATASET,WORD2VEC_MODEL_FILE, MAX_LEN_FILE)
    val_dataset = CommentDataset(VAL_DATASET, WORD2VEC_MODEL_FILE, MAX_LEN_FILE)
    test_dataset = CommentDataset(TEST_DATASET, WORD2VEC_MODEL_FILE, MAX_LEN_FILE)

    # print(train_dataset[540],train_dataset[541],train_dataset[542])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=min(os.cpu_count(), batch_size), shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=min(os.cpu_count(), batch_size), shuffle=False, pin_memory=True)
    test_loadder = DataLoader(test_dataset, batch_size=64, num_workers=min(os.cpu_count(), batch_size), shuffle=False, pin_memory=True)

    dtype1 = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    dtype2 = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    for epoch in range(start_epoch, epochs):
        model.train()
        if epoch > 0:
            scheduler.step()
        train_loss = 0 
        train_acc = 0    
        pbar = tqdm(enumerate(train_loader), total=len(train_dataset))
        for i, (sentences, labels) in pbar:
            sentences = sentences.type(dtype1).to(device)
            labels = labels.type(dtype2).to(device)
            preds = model(sentences)
            loss = criterion(preds, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += (preds.argmax(1) == labels).sum().item()

        print("Epoch: {} || Train Loss: {:.3%f} || Train Acc: {:.3%f} ".format(epoch+1,train_loss/len(train_dataset), train_acc/len(train_dataset)))
    
        model.eval()
        val_loss = 0
        val_acc = 0
        for (sentences,labels) in val_loader:
            sentences = sentences.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                preds = model(sentences)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                val_acc += (preds.argmax(1) == labels).sum().item()
        if best_fitness < val_acc:
            best_fitness = val_acc
            chkpt = {'epoch':epoch, 
                     'best_fitness':best_fitness,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()
                    }
            torch.save(chkpt,best_weight)

            del chkpt

    torch.cuda.empty_cache()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=bool, default=True, help='use momentum in SGD')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay in SGD')
    parser.add_argument('--resume', action='store_true', help='resume training')
    opts = parser.parse_args()
    print(opts)
    train(opts)
    
