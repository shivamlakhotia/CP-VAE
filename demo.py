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
from models.vae import VAE

import random

class Demo:
    def __init__(self, test, test_labels1, test_labels2, load_path, vocab, vae_params):
        super(Demo, self).__init__()

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_path = load_path

        self.vocab = vocab
        self.test_data = test
        self.test_labels1 = test_labels1
        self.test_labels2 = test_labels2

        self.vae = VAE(**vae_params)
        if self.use_cuda:
            self.vae.cuda()

        self.nbatch = len(self.test_data)
        self.load(self.load_path)
    
    def load(self, path):
        model_path = os.path.join(path, "model.pt")
        self.vae.load_state_dict(torch.load(model_path))
    
    def stop_train_mean_embeddings(self, num_pos_embeddings, num_neg_embeddings, num_past_embeddings, num_present_embeddings, max_embeddings):
        return (num_pos_embeddings == max_embeddings) and ((num_neg_embeddings == max_embeddings)) and ((num_past_embeddings == max_embeddings)) and ((num_present_embeddings == max_embeddings))

    def train_mean_embeddings(self):
        self.vae.eval()
        max_embeddings = 100
        num_pos_embeddings = 0
        num_neg_embeddings = 0
        num_past_embeddings = 0
        num_present_embeddings = 0
        mu_pos = None
        mu_neg = None
        mu_past = None
        mu_present = None
        var_pos = None
        var_neg = None
        var_past = None
        var_present = None
        with torch.no_grad():
            while True:
                if self.stop_train_mean_embeddings(num_pos_embeddings, num_neg_embeddings, num_past_embeddings, num_present_embeddings, max_embeddings):
                    break
                batch_idx = random.randint(5, self.nbatch-1)
                sent_idx = random.randint(0, self.test_data[batch_idx].size()[1]-1)

                mu_c, logvar_c, mu_s1, logvar_s1, mu_s2, logvar_s2 = self.vae.encoder(self.test_data[batch_idx][:, sent_idx:sent_idx+1])
                assert (mu_c.size()[0] == 1)

                if self.test_labels1[batch_idx][sent_idx] == 0 and num_neg_embeddings < max_embeddings:
                    num_neg_embeddings += 1
                    if mu_neg is None:
                        assert (var_neg is None)
                        mu_neg = mu_s1
                        var_neg = logvar_s1.exp()
                    else:
                        mu_neg += mu_s1
                        var_neg += logvar_s1.exp()

                if self.test_labels1[batch_idx][sent_idx] == 1 and num_pos_embeddings < max_embeddings:
                    num_pos_embeddings += 1
                    if mu_pos is None:
                        assert (var_pos is None)
                        mu_pos = mu_s1
                        var_pos = logvar_s1.exp()
                    else:
                        mu_pos += mu_s1
                        var_pos += logvar_s1.exp()
                
                if self.test_labels2[batch_idx][sent_idx] == 0 and num_past_embeddings < max_embeddings:
                    num_past_embeddings += 1
                    if mu_past is None:
                        assert (var_past is None)
                        mu_past = mu_s2
                        var_past = logvar_s2.exp()
                    else:
                        mu_past += mu_s2
                        var_past += logvar_s2.exp()
                
                if self.test_labels2[batch_idx][sent_idx] == 1 and num_present_embeddings < max_embeddings:
                    num_present_embeddings += 1
                    if mu_present is None:
                        assert (var_present is None)
                        mu_present = mu_s2
                        var_present = logvar_s2.exp()
                    else:
                        mu_present += mu_s2
                        var_present += logvar_s2.exp()
        
        return mu_pos/max_embeddings, (var_pos/max_embeddings**2).log(), mu_neg/max_embeddings, (var_neg/max_embeddings**2).log(), \
            mu_past/max_embeddings, (var_past/max_embeddings**2).log(), mu_present/max_embeddings, (var_present/max_embeddings**2).log()

    def train_and_save_embeddings(self):
        mu_pos, logvar_pos, mu_neg, logvar_neg, mu_past, logvar_past, mu_present, logvar_present = self.train_mean_embeddings()

        np.savez('data/demo_embeddings.npz', mu_pos=mu_pos.cpu().numpy(), logvar_pos=logvar_pos.cpu().numpy(), mu_neg=mu_neg.cpu().numpy(), logvar_neg=logvar_neg.cpu().numpy(), \
            mu_past=mu_past.cpu().numpy(), logvar_past=logvar_past.cpu().numpy(), mu_present=mu_present.cpu().numpy(), logvar_present=logvar_present.cpu().numpy())


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

    demo = Demo(train_data, train_sentiments, train_tenses, args.load_path, vocab, params["vae_params"])
    demo.train_and_save_embeddings()

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp',
                        help='data name')
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