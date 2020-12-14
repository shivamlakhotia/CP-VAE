# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd
import argparse
from utils.bleu import compute_bleu
import os

def main(args):
    data_pth = "data/generated_eval_data"
    file_name = args.model_name+"_"+args.swap_type+"_swapped_file.csv"
    file_pth = os.path.join(data_pth, file_name)
    data_file = pd.read_csv(file_pth, sep='\t').drop(axis = 1, columns=['Unnamed: 0'])
    data_file['Original'] = data_file['Original'].str[3:-5]
    data_file['Swapped'] = data_file['Swapped'].str[3:]

    # BLEU Score
    total_bleu = 0.0
    sources = []
    targets = []
    for i in range(data_file.shape[0]):
        s = data_file['Original'][i].split()
        t = data_file['Swapped'][i].split()
        sources.append([s])
        targets.append(t)

    total_bleu += compute_bleu(sources, targets)[0]
    total_bleu *= 100
    print("Bleu: %.2f" % total_bleu)


def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--swap_type', type=str, default='both',help='one out of [both,sentiment,tense]')
    parser.add_argument('--model_name', type=str, default='M3',help='one out of [M1,M2,M3]')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
