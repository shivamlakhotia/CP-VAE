# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from utils.text_utils import MonoTextData
import numpy as np
import os
import random
from classifier import CNNClassifier


def evaluate(model, eval_data, eval_label):
    correct_num = 0
    total_sample = 0
    i = 0
    for batch_data, batch_label in zip(eval_data, eval_label):
        print("Evaluating batch = ",i)
        batch_size = batch_data.size(0)
        logits = model(batch_data)
        probs = torch.sigmoid(logits)
        y_hat = list((probs > 0.5).long().cpu().numpy())
        correct_num += sum([p == q for p, q in zip(batch_label, y_hat)])
        total_sample += batch_size
        i = i + 1
    return correct_num / total_sample

def main(args):
    data_pth = "results/%s" % args.data_name
    train_pth = os.path.join(data_pth, ("train_identical_{}_{}.txt").format(str(args.confidence+10),args.style))
    #dev_pth = os.path.join(data_pth, "dev_identical_80_%s.txt" % args.style)
    test_pth = os.path.join(data_pth, ("test_identical_{}_{}.txt").format(str(args.confidence+10),args.style))

    train_data = MonoTextData(train_pth, True, vocab=100000)
    #random.shuffle(train_data.data)

    vocab = train_data.vocab
    #dev_data = MonoTextData(dev_pth, True, vocab=vocab)
    #random.shuffle(dev_data.data)
    test_data = MonoTextData(test_pth, True, vocab=vocab)
    path = "checkpoint/{}-identical-{}-{}-classifier.pt".format(str(args.confidence),args.data_name,args.style)
    #path = "checkpoint/%s-classifier.pt" % args.data_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #train_batch, train_label = train_data.create_data_batch_labels(64, device, batch_first=True)
    #dev_batch, dev_label = dev_data.create_data_batch_labels(64, device, batch_first=True)
    test_batch, test_label = test_data.create_data_batch_labels(64, device, batch_first=True)

    #nbatch = len(train_batch)
    #best_acc = 0.0
    #step = 0

    checkpoint = torch.load(path)
    model = CNNClassifier(len(checkpoint['embedding.weight']), 300, [1,2,3,4,5], 500, 0.5).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        acc = evaluate(model, test_batch, test_label)
    print('Test Acc: %.2f' % acc)

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--style', type=str, default='sentiment')
    parser.add_argument('--confidence',type=int,default=80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
