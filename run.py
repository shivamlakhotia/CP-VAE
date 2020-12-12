# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils.exp_utils import create_exp_dir
from utils.text_utils import MonoTextData
import argparse
import os
import torch
import time
import config
from models.decomposed_vae import DecomposedVAE
import numpy as np
import logging

def main(args):
    conf = config.CONFIG[args.data_name]
    data_pth = "data/%s" % args.data_name


    train_sentiment_data_pth = os.path.join(data_pth, "train_sentiment_data.txt")
    train_sentiment_feat_pth = os.path.join(data_pth, "train_sentiment_%s.npy" % args.feat)
    train_sentiment_data = MonoTextData(train_sentiment_data_pth, True)
    train_sentiment_feat = np.load(train_sentiment_feat_pth)

    train_tense_data_pth = os.path.join(data_pth, "train_tense_data.txt")
    train_tense_feat_pth = os.path.join(data_pth, "train_tense_%s.npy" % args.feat)
    train_tense_data = MonoTextData(train_tense_data_pth, True)
    train_tense_feat = np.load(train_tense_feat_pth)


    sentiment_vocab = train_sentiment_data.vocab
    print('Sentiment Vocabulary size: %d' % len(sentiment_vocab))

    tense_vocab = train_tense_data.vocab
    print('Tense Vocabulary size: %d' % len(tense_vocab))

    dev_sentiment_data_pth = os.path.join(data_pth, "dev_sentiment_data.txt")
    dev_sentiment_feat_pth = os.path.join(data_pth, "dev_sentiment_%s.npy" % args.feat)
    dev_sentiment_data = MonoTextData(dev_sentiment_data_pth, True, vocab=sentiment_vocab)
    dev_sentiment_feat = np.load(dev_sentiment_feat_pth)

    dev_tense_data_pth = os.path.join(data_pth, "dev_tense_data.txt")
    dev_tense_feat_pth = os.path.join(data_pth, "dev_tense_%s.npy" % args.feat)
    dev_tense_data = MonoTextData(dev_tense_data_pth, True, vocab=tense_vocab)
    dev_tense_feat = np.load(dev_tense_feat_pth)


    test_sentiment_data_pth = os.path.join(data_pth, "test_sentiment_data.txt")
    test_sentiment_feat_pth = os.path.join(data_pth, "test_sentiment_%s.npy" % args.feat)
    test_sentiment_data = MonoTextData(test_sentiment_data_pth, True, vocab=sentiment_vocab)
    test_sentiment_feat = np.load(test_sentiment_feat_pth)

    test_tense_data_pth = os.path.join(data_pth, "test_tense_data.txt")
    test_tense_feat_pth = os.path.join(data_pth, "test_tense_%s.npy" % args.feat)
    test_tense_data = MonoTextData(test_tense_data_pth, True, vocab=tense_vocab)
    test_tense_feat = np.load(test_tense_feat_pth)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path0 = 'sentiment-{}-{}-{}'.format(args.save, args.data_name, args.feat)
    save_path0 = os.path.join(save_path0, time.strftime("%Y%m%d-%H%M%S"))
    save_path1 = 'tense-{}-{}-{}'.format(args.save, args.data_name, args.feat)
    save_path1 = os.path.join(save_path1, time.strftime("%Y%m%d-%H%M%S"))

    scripts_to_save = [
        'run.py', 'models/decomposed_vae.py', 'models/vae.py',
        'models/base_network.py', 'config.py']
    logging0 = create_exp_dir(save_path0, scripts_to_save=scripts_to_save,
                             debug=args.debug)
    logging1 = create_exp_dir(save_path1, scripts_to_save=scripts_to_save,
                             debug=args.debug)

    if args.text_only:
        train_sentiment = train_sentiment_data.create_data_batch(args.bsz, device)
        dev_sentiment = dev_sentiment_data.create_data_batch(args.bsz, device)
        test_sentiment = test_sentiment_data.create_data_batch(args.bsz, device)
        feat_sentiment = train_sentiment

        train_tense = train_tense_data.create_data_batch(args.bsz, device)
        test_tense = test_tense_data.create_data_batch(args.bsz, device)
        feat_tense = train_tense
    else:
        train_sentiment = train_sentiment_data.create_data_batch_feats(args.bsz, train_sentiment_feat, device)
        dev_sentiment = dev_sentiment_data.create_data_batch_feats(args.bsz, dev_sentiment_feat, device)
        test_sentiment = test_sentiment_data.create_data_batch_feats(args.bsz, test_sentiment_feat, device)
        feat_sentiment = train_sentiment_feat
        train_tense = train_tense_data.create_data_batch_feats(args.bsz, train_tense_feat, device)
        test_tense = test_tense_data.create_data_batch_feats(args.bsz, test_tense_feat, device)
        feat_tense = train_tense_feat

    #VAE training on sentiment data
    # kwargs0 = {
    #     "train": train_sentiment,
    #     "valid": dev_sentiment,
    #     "test": test_sentiment,
    #     "feat": feat_sentiment,
    #     "bsz": args.bsz,
    #     "save_path": save_path0,
    #     "logging": logging0,
    #     "text_only": args.text_only,
    # }
    # params = conf["params"]
    # params["vae_params"]["vocab"] = sentiment_vocab
    # params["vae_params"]["device"] = device
    # params["vae_params"]["text_only"] = args.text_only
    # params["vae_params"]["mlp_ni"] = train_sentiment_feat.shape[1]
    # kwargs0 = dict(kwargs0, **params)

    # sentiment_model = DecomposedVAE(**kwargs0)
    # try:
    #     valid_loss = sentiment_model.fit()
    #     logging("sentiment val loss : {}".format(valid_loss))
    # except KeyboardInterrupt:
    #     logging("Exiting from training early")

    # sentiment_model.load(save_path0)
    # test_loss = model.evaluate(sentiment_model.test_data, sentiment_model.test_feat)
    # logging("sentiment test loss: {}".format(test_loss[0]))
    # logging("sentiment test recon: {}".format(test_loss[1]))
    # logging("sentiment test kl1: {}".format(test_loss[2]))
    # logging("sentiment test kl2: {}".format(test_loss[3]))
    # logging("sentiment test mi1: {}".format(test_loss[4]))
    # logging("sentiment test mi2: {}".format(test_loss[5]))

    #VAE training on tense data
    kwargs1 = {
        "train": train_tense,
        "valid": test_tense,
        "test": test_tense,
        "feat": feat_tense,
        "bsz": args.bsz,
        "save_path": save_path1,
        "logging": logging1,
        "text_only": args.text_only,
    }
    params = conf["params"]
    params["vae_params"]["vocab"] = tense_vocab
    params["vae_params"]["device"] = device
    params["vae_params"]["text_only"] = args.text_only
    params["vae_params"]["mlp_ni"] = train_tense_feat.shape[1]
    kwargs1 = dict(kwargs1, **params)

    tense_model = DecomposedVAE(**kwargs1)
    try:
        valid_loss = tense_model.fit()
        logging("tense val loss : {}".format(valid_loss))
    except KeyboardInterrupt:
        logging("Exiting from training early")

    tense_model.load(save_path1)
    test_loss = model.evaluate(tense_model.test_data, tense_model.test_feat)
    logging("tense test loss: {}".format(test_loss[0]))
    logging("tense test recon: {}".format(test_loss[1]))
    logging("tense test kl1: {}".format(test_loss[2]))
    logging("tense test kl2: {}".format(test_loss[3]))
    logging("tense test mi1: {}".format(test_loss[4]))
    logging("tense test mi2: {}".format(test_loss[5]))


def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp',
                        help='data name')
    parser.add_argument('--save', type=str, default='checkpoint/ours',
                        help='directory name to save')
    parser.add_argument('--bsz', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--text_only', default=False, action='store_true',
                        help='use text only without feats')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='enable debug mode')
    parser.add_argument('--feat', type=str, default='glove',
                        help='feat repr')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
