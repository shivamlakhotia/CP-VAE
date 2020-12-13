# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd
import argparse
from utils.bleu import compute_bleu
from utils.text_utils import MonoTextData
import torch
from classifier import CNNClassifier, evaluate
import os

def main(args):
    data_pth = "data"
    file_pth = os.path.join(data_pth, args.file_name)
    data_file = pd.read_csv(file_pth, sep='\t').drop(axis = 1, columns=['Unnamed: 0'])
    data_file['Original'] = data_file['Original'].str[3:-5]
    data_file['Swapped'] = data_file['Swapped'].str[3:]
    data_file['New Sentiment Label'] = data_file['New Sentiment Label'].astype('int')
    data_file['New Tense Label'] = data_file['New Tense Label'].astype('int')

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sentiment_label = data_file['New Sentiment Label']
    tense_label = data_file['New Tense Label']
    torch_sent_label = torch.tensor(sentiment_label, device=device, requires_grad=False)
    torch_tense_label = torch.tensor(tense_label, device=device, requires_grad=False)

    train_data = MonoTextData(train_pth, True, vocab=100000)
    vocab = train_data.vocab
    source_pth = os.path.join(data_pth, "test_identical_sentiment_90_tense.csv")
    target_pth = args.target_path
    eval_data = MonoTextData(target_pth, True, vocab=vocab)

    # Classification Accuracy

    eval_data, eval_sent_label, eval_tense_label = eval_data.create_data_batch_labels(64, device, batch_first=True)

    model_sent = CNNClassifier(len(vocab), 300, [1, 2, 3, 4, 5], 500, 0.5).to(device)
    model_sent.load_state_dict(torch.load("checkpoint/our-%s/%s-sentiment-classifier.pt" % (args.data_name,args.data_name)))
    model_sent.eval()
    acc_sent = 100 * evaluate(model_sent, eval_data, eval_sent_label)
    print("Sentiment Acc: %.2f" % acc_sent)
    
    model_tense = CNNClassifier(len(vocab), 300, [1, 2, 3, 4, 5], 500, 0.5).to(device)
    model_tense.load_state_dict(torch.load("checkpoint/our-%s/%s-tense-classifier.pt" % (args.data_name,args.data_name)))
    model_tense.eval()
    acc_tense = 100 * evaluate(model_tense, eval_data, eval_tense_label)
    print("Tense Acc: %.2f" % acc_tense)

    # BLEU Score
    total_bleu = 0.0
    sources = []
    targets = []
    for i in range(source.shape[0]):
        s = source.content[i].split()
        t = target.content[i].split()
        sources.append([s])
        targets.append(t)

    total_bleu += compute_bleu(sources, targets)[0]
    total_bleu *= 100
    print("Bleu: %.2f" % total_bleu)

def add_args(parser):
    parser.add_argument('--file_name', type=str, default='amol_both_swapped_file.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
