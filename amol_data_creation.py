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
import pandas as pd

import random

class AmolData:
    def __init__(self, test, test_labels1, test_labels2, load_path, vocab, vae_params):
        super(AmolData, self).__init__()

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
        self.load_embeddings()
    
    def load(self, path):
        model_path = os.path.join(path, "model.pt")
        self.vae.load_state_dict(torch.load(model_path))
    
    def load_embeddings(self):
        data = np.load('data/demo_embeddings.npz')

        self.mu_pos = torch.tensor(data['mu_pos']).unsqueeze(0).to(self.device)
        self.logvar_pos = torch.tensor(data['logvar_pos']).unsqueeze(0).to(self.device)
        self.mu_neg = torch.tensor(data['mu_neg']).unsqueeze(0).to(self.device)
        self.logvar_neg = torch.tensor(data['logvar_neg']).unsqueeze(0).to(self.device)
        self.mu_past = torch.tensor(data['mu_past']).unsqueeze(0).to(self.device)
        self.logvar_past = torch.tensor(data['logvar_past']).unsqueeze(0).to(self.device)
        self.mu_present = torch.tensor(data['mu_present']).unsqueeze(0).to(self.device)
        self.logvar_present = torch.tensor(data['logvar_present']).unsqueeze(0).to(self.device)

    def create_data(self):
        self.vae.eval()
        total_sent = 0
        df_original = []
        df_sentiment_swapped = []
        df_sentiment_swapped_l1 = []
        df_sentiment_swapped_l2 = []

        df_tense_swapped = []
        df_tense_swapped_l1 = []
        df_tense_swapped_l2 = []
        df_both_swapped = []
        df_both_swapped_l1 = []
        df_both_swapped_l2 = []

        with torch.no_grad():
            while total_sent < 1000:
                total_sent += 1
                batch_idx = random.randint(0, self.nbatch-1)
                sent_idx = random.randint(0, self.test_data[batch_idx].size()[1]-1)
                mu_c, logvar_c, mu_s1, logvar_s1, mu_s2, logvar_s2 = self.vae.encoder(self.test_data[batch_idx][:, sent_idx:sent_idx+1])
                original_sentence = ""

                for j in range(self.test_data[batch_idx].size()[0]):
                    original_sentence += self.vocab.id2word(self.test_data[batch_idx][j, sent_idx:sent_idx+1]) + " "
                
                original_sentiment = self.test_labels1[batch_idx][sent_idx]
                original_tense = self.test_labels2[batch_idx][sent_idx]
                print("original_sentence:", original_sentence, "sentiment:", original_sentiment, "tense:", original_tense)

                c, s1, s2, _ = self.vae.encode(self.test_data[batch_idx][:, sent_idx:sent_idx+1])

                if original_sentiment == 0:
                    transfer_sentiment = self.vae.decoder.beam_search_decode(c, self.mu_pos, s2)
                    sentiment_swap_l1 = 1
                    sentiment_swap_l2 = original_tense.item()
                elif original_sentiment == 1:
                    transfer_sentiment = self.vae.decoder.beam_search_decode(c, self.mu_neg, s2)
                    sentiment_swap_l1 = 0
                    sentiment_swap_l2 = original_tense.item()

                if original_tense == 0:
                    transfer_tense = self.vae.decoder.beam_search_decode(c, s1, self.mu_present)
                    tense_swap_l1 = original_sentiment.item()
                    tense_swap_l2 = 1
                elif original_tense == 1:
                    transfer_tense = self.vae.decoder.beam_search_decode(c, s1, self.mu_past)
                    tense_swap_l1 = original_sentiment.item()
                    tense_swap_l2 = 0
                
                if original_sentiment == 0 and original_tense == 0:
                    transfer_both = self.vae.decoder.beam_search_decode(c, self.mu_pos, self.mu_present)
                    both_swap_l1 = 1
                    both_swap_l2 = 1
                elif original_sentiment == 0 and original_tense == 1:
                    transfer_both = self.vae.decoder.beam_search_decode(c, self.mu_pos, self.mu_past)
                    both_swap_l1 = 1
                    both_swap_l2 = 0
                elif original_sentiment == 1 and original_tense == 0:
                    transfer_both = self.vae.decoder.beam_search_decode(c, self.mu_neg, self.mu_present)
                    both_swap_l1 = 0
                    both_swap_l2 = 1
                elif original_sentiment == 1 and original_tense == 1:
                    transfer_both = self.vae.decoder.beam_search_decode(c, self.mu_neg, self.mu_past)
                    both_swap_l1 = 0
                    both_swap_l2 = 0

                df_original.append(original_sentence)
                df_sentiment_swapped.append(" ".join(transfer_sentiment[0][:-1]))
                df_tense_swapped.append(" ".join(transfer_tense[0][:-1]))
                df_both_swapped.append(" ".join(transfer_both[0][:-1]))
                df_sentiment_swapped_l1.append(sentiment_swap_l1)
                df_sentiment_swapped_l2.append(sentiment_swap_l2)
                df_tense_swapped_l1.append(tense_swap_l1)
                df_tense_swapped_l2.append(tense_swap_l2)
                df_both_swapped_l1.append(both_swap_l1)
                df_both_swapped_l2.append(both_swap_l2)
            
        pd.options.display.max_colwidth = 100
        df_sentiment_swap_file = pd.DataFrame()
        df_sentiment_swap_file["Original"] = df_original
        df_sentiment_swap_file["Swapped"] = df_sentiment_swapped
        df_sentiment_swap_file["New Sentiment Label"] = df_sentiment_swapped_l1
        df_sentiment_swap_file["New Tense Label"] = df_sentiment_swapped_l2
        df_sentiment_swap_file.to_csv('data/generated_eval_data_1/M1_sentiment_swapped_file.csv', sep='\t')
  
        df_tense_swap_file = pd.DataFrame()
        df_tense_swap_file["Original"] = df_original
        df_tense_swap_file["Swapped"] = df_tense_swapped
        df_tense_swap_file["New Sentiment Label"] = df_tense_swapped_l1
        df_tense_swap_file["New Tense Label"] = df_tense_swapped_l2
        df_tense_swap_file.to_csv('data/generated_eval_data_1/M1_tense_swapped_file.csv', sep='\t')

        df_both_swap_file = pd.DataFrame()
        df_both_swap_file["Original"] = df_original
        df_both_swap_file["Swapped"] = df_both_swapped
        df_both_swap_file["New Sentiment Label"] = df_both_swapped_l1
        df_both_swap_file["New Tense Label"] = df_both_swapped_l2
        df_both_swap_file.to_csv('data/generated_eval_data_1/M1_both_swapped_file.csv', sep='\t')

        # print(df)


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

    test_data_pth = os.path.join(data_pth, "test_identical_sentiment_90_tense.csv")
    test_class = MonoTextData(test_data_pth, vocab=vocab, glove=True)
    test_data, test_sentiments, test_tenses = test_class.create_data_batch_labels(args.bsz, device)

    print("data done.")

    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device

    amolData = AmolData(test_data, test_sentiments, test_tenses, args.load_path, vocab, params["vae_params"])
    amolData.create_data()

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp',
                        help='data name')
    parser.add_argument('--load_path', type=str, default='',
                        help='directory name to load')
    parser.add_argument('--bsz', type=int, default=256,
                        help='batch size for training')
    parser.add_argument('--vocab', type=str, default='./tmp/yelp.vocab')
    parser.add_argument('--embedding', type=str, default='./data/glove.840B.300d.txt')
    parser.add_argument('--dim_emb', type=int, default=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)