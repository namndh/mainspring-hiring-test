import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from model.model import myModel
from dataset import CommentDataset
from config import *


def test():

    wdir = os.path.join(ROOT_DIR, 'weights')
    best_weight = os.path.join(wdir, 'best.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = myModel()
    model.double()
    model.to(device)

    assert os.path.isdir(wdir), 'weight folder does not exist!'
    assert os.path.isfile(
        best_weight), 'the best weight file does not exist!'
    try:
        chkpt = torch.load(best_weight, map_location=device)
        print('best_fitness',chkpt['epoch'])
        model.load_state_dict(chkpt['model'])

        if chkpt.get('training_results') is not None:
            print(chkpt['training_results'])
        del chkpt
    except Exception:
        SystemExit('Error: Can not load pretrained model!')

    test_dataset = CommentDataset(
        TEST_DATASET, WORD2VEC_MODEL_FILE, MAX_LEN_FILE)


    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=min(
        os.cpu_count(), 64), shuffle=False)

    
    print('\n\nTesting...')
    model.eval()
    test_acc = 0
    y_true = []
    y_pred = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, (sentences, labels) in pbar:
        sentences = sentences.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(sentences)

            test_acc += (preds.argmax(1) == labels).sum().item()
            y_pred += preds.argmax(1).to('cpu').data.numpy().tolist()
            y_true += labels.to('cpu').data.numpy().tolist()

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    s_test = "Acc: {:.3f}% || Precision: {:.3f} || Recall: {:.3f} || F1-score: {:.3f}".format(test_acc/len(
        test_dataset)*100, precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred))
    print(s_test)
    with open('test_results.txt', 'w+') as f:
        f.write(s_test)

    torch.cuda.empty_cache()


if __name__ == '__main__':

    test()
