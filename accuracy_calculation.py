# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd
import argparse
from utils.text_utils import MonoTextData
import torch
from classifier import CNNClassifier, evaluate
import os

def main(args):
    data_pth = "data/generated_eval_data_1"
    file_name = "acc_"+args.model_name+"_"+args.swap_type+"_swapped_file.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_pth = os.path.join("data", "%s/train_identical_sentiment_90_tense.csv"%args.data_name)
    train_data = MonoTextData(train_pth, True, vocab=100000)
    vocab = train_data.vocab

    bleu_file_pth = os.path.join(data_pth, file_name)
    eval_data = MonoTextData(bleu_file_pth, True,vocab = vocab)

    # Classification Accuracy

    eval_data, eval_sent_label, eval_tense_label = eval_data.create_data_batch_labels(64, device, batch_first=True)

    model_sent = CNNClassifier(len(vocab), 300, [1, 2, 3, 4, 5], 500, 0.5).to(device)
    model_sent.load_state_dict(torch.load("checkpoint/ours-%s/%s-sentiment-classifier.pt" % (args.data_name,args.data_name)))
    model_sent.eval()
    acc_sent = 100 * evaluate(model_sent, eval_data, eval_sent_label)
    print("Sent Acc: %.2f" % acc_sent)
    
    model_tense = CNNClassifier(len(vocab), 300, [1, 2, 3, 4, 5], 500, 0.5).to(device)
    model_tense.load_state_dict(torch.load("checkpoint/ours-%s/%s-tense-classifier.pt" % (args.data_name,args.data_name)))
    model_tense.eval()
    acc_tense = 100 * evaluate(model_tense, eval_data, eval_tense_label)
    print("Tense Acc: %.2f" % acc_tense)


def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--swap_type', type=str, default='both',help='one out of [both,sentiment,tense]')
    parser.add_argument('--model_name', type=str, default='M3',help='one out of [M1,M2,M3]')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
