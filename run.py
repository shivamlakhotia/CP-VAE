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
from models.vae import TrainerVAE

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)

    conf = config.CONFIG[args.data_name] # Need to update !!
    data_pth = "data/%s" % args.data_name
    train_data_pth = os.path.join(data_pth, "train_identical_sentiment_90_tense.csv")
    train_class = MonoTextData(train_data_pth, glove=True)
    train_data, train_sentiments, train_tenses = train_class.create_data_batch_labels(args.bsz, device)

    vocab = train_class.vocab
    print('Vocabulary size: %d' % len(vocab))

    val_data_pth = os.path.join(data_pth, "dev_identical_sentiment_90_tense.csv")
    val_class = MonoTextData(val_data_pth, vocab=vocab, glove=True)
    val_data, val_sentiments, val_tenses = val_class.create_data_batch_labels(args.bsz, device)

    test_data_pth = os.path.join(data_pth, "test_identical_sentiment_90_tense.csv")
    test_class = MonoTextData(test_data_pth, vocab=vocab, glove=True)
    test_data, test_sentiments, test_tenses = test_class.create_data_batch_labels(args.bsz, device)

    print("data done.")

    save_path = '{}-{}'.format(args.save, args.data_name)
    folder_name = time.strftime("%Y%m%d-%H%M%S") + '-' + args.exp_name
    save_path = os.path.join(save_path, folder_name)
    scripts_to_save = [
        'run.py', 'models/vae.py',
        'models/base_network.py', 'config.py']
    logging = create_exp_dir(save_path, scripts_to_save=scripts_to_save,
                             debug=args.debug)

    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device

    trainerVAE = TrainerVAE(train_data, val_data, test_data, train_sentiments, train_tenses, val_sentiments, val_tenses, test_sentiments, test_tenses, save_path, logging, 
    params["num_epochs"], params["log_interval"], params["warm_up"], params["kl_start"], params["vae_params"], params["lr_params"], args.dist_train)

    trainerVAE.fit()


def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp',
                        help='data name')
    parser.add_argument('--save', type=str, default='checkpoint/ours',
                        help='directory name to save')
    parser.add_argument('--exp_name', type=str, default='',
                        help='experiment Name')
    parser.add_argument('--bsz', type=int, default=256,
                        help='batch size for training')
    parser.add_argument('--text_only', default=False, action='store_true',
                        help='use text only without feats')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='enable debug mode')
    parser.add_argument('--feat', type=str, default='glove',
                        help='feat repr')
    parser.add_argument('--vocab', type=str, default='./tmp/yelp.vocab')
    parser.add_argument('--embedding', type=str, default='./data/glove.840B.300d.txt')
    parser.add_argument('--dim_emb', type=int, default=300)
    parser.add_argument('--dist_train', default=False, action='store_true',
                        help='want distribution train loss?')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
