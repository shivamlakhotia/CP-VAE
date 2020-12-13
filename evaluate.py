from utils.exp_utils import create_exp_dir
from utils.text_utils import MonoTextData
import argparse
import os
import torch
import time
import config
# from models.decomposed_vae import DecomposedVAE
import numpy as np
from file_io import *
from vocab import Vocabulary, build_vocab
from models.vae import TrainerVAE, EvaluateVAE

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # np.random.seed(0)
    # torch.manual_seed(0)

    conf = config.CONFIG[args.data_name] # Need to update !!
    data_pth = "data/%s" % args.data_name
    train_data_pth = os.path.join(data_pth, "train_identical_sentiment_90_tense.csv")
    train_class = MonoTextData(train_data_pth, glove=True)
    train_data, train_sentiments, train_tenses = train_class.create_data_batch_labels(args.bsz, device)

    vocab = train_class.vocab
    print('Vocabulary size: %d' % len(vocab))

    test_data_pth = os.path.join(data_pth, "eval_data.csv")
    test_class = MonoTextData(test_data_pth, vocab=vocab, glove=True)
    test_data, test_sentiments, test_tenses = test_class.create_data_batch_labels(args.bsz, device)

    print("data done.")

    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device

    evalVAE = EvaluateVAE(test_data, test_sentiments, test_tenses, args.load_path, vocab, params["vae_params"])
    evalVAE.eval_style_transfer()

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp',
                        help='data name')
    parser.add_argument('--train', type=str, default='./data/yelp/sentiment.train',
                        help='train data path')
    parser.add_argument('--dev', type=str, default='./data/yelp/sentiment.dev',
                        help='val data path')
    parser.add_argument('--test', type=str, default='./data/yelp/sentiment.test',
                        help='test data path')
    parser.add_argument('--load_path', type=str, default='',
                        help='directory name to load')
    parser.add_argument('--bsz', type=int, default=128,
                        help='batch size for training')
    parser.add_argument('--vocab', type=str, default='./tmp/yelp.vocab')
    parser.add_argument('--embedding', type=str, default='./data/glove.840B.300d.txt')
    parser.add_argument('--dim_emb', type=int, default=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)