import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# from sklearn.metrics.pairwise import pairwise_distances
# from sklearn.manifold.t_sne import (_joint_probabilities,
#                                     _kl_divergence)
# from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
# % matplotlib inline

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

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

class TSNE_data:
    def __init__(self, test, test_labels1, test_labels2, load_path, vocab, vae_params):
        super(TSNE_data, self).__init__()

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
        self.batch_size = len(self.test_data[0])
        self.load(self.load_path)
    
    def load(self, path):
        model_path = os.path.join(path, "model.pt")
        self.vae.load_state_dict(torch.load(model_path))

    def get_embeddings(self):
        self.vae.eval()
        s1_embeddings = None
        s2_embeddings = None
        c_embeddings = None
        label_sorted = None
        labels1 = None
        labels2 = None
        embeddings_bool = False

        with torch.no_grad():
            for idx, batch_data in enumerate(self.test_data):
                if idx > 400 and idx < 430:
                    print("dataset_idx:", idx)
                    sent_len, batch_size = batch_data.size()
                    batch_labels1 = self.test_labels1[idx]
                    batch_labels2 = self.test_labels2[idx]

                    batch_labels_int = 2*batch_labels1 + batch_labels2

                    # c, s1, s2, _ = self.vae.encode(batch_data)

                    # n_sample_c, batch_size_c, nc = c.size()
                    # n_sample_s1, batch_size_s1, ns1 = s1.size()
                    # n_sample_s2, batch_size_s2, ns2 = s2.size()

                    # c = c.view(batch_size_c * n_sample_c, nc)
                    # s1 = s1.view(batch_size_s1 * n_sample_s1, ns1)
                    # s2 = s2.view(batch_size_s2 * n_sample_s2, ns2)

                    c, _, s1, _, s2, _ = self.vae.encoder(batch_data)

                    if not embeddings_bool:
                        s1_embeddings = s1
                        s2_embeddings = s2
                        c_embeddings = c
                        label_sorted = batch_labels_int
                        labels1 = batch_labels1
                        labels2 = batch_labels2
                        embeddings_bool = True
                    else:
                        s1_embeddings = torch.cat((s1_embeddings, s1), 0)
                        s2_embeddings = torch.cat((s2_embeddings, s2), 0)
                        c_embeddings = torch.cat((c_embeddings, c), 0)
                        label_sorted = torch.cat((label_sorted, batch_labels_int), 0)
                        labels1 = torch.cat((labels1, batch_labels1), 0)
                        labels2 = torch.cat((labels2, batch_labels2), 0)
                        # label_sorted = torch.cat((label_sorted, batch_labels_int), 0)
                    
        
        return c_embeddings.cpu().numpy(), s1_embeddings.cpu().numpy(), s2_embeddings.cpu().numpy(), label_sorted.cpu().numpy(), labels1.cpu().numpy(), labels2.cpu().numpy()


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 4))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=20,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(4):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

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

    tsne_data = TSNE_data(train_data, train_sentiments, train_tenses, args.load_path, vocab, params["vae_params"])

    c_embeddings, s1_embeddings, s2_embeddings, labels, labels1, labels2 = tsne_data.get_embeddings()
    print("Embeddings Formed!!", np.shape(np.array(s1_embeddings)))

    s1_proj = TSNE(random_state=RS).fit_transform(s1_embeddings)
    s2_proj = TSNE(random_state=RS).fit_transform(s2_embeddings)
    c_proj = TSNE(random_state=RS).fit_transform(c_embeddings)
    print("TSNE projections done!!")

    scatter(s1_proj, labels1)
    plt.savefig('data/s1_embeddings', dpi=120)
    scatter(s2_proj, labels2)
    plt.savefig('data/s2_embeddings', dpi=120)
    scatter(s1_proj, labels2)
    plt.savefig('data/s1_label2_embeddings', dpi=120)
    scatter(s2_proj, labels1)
    plt.savefig('data/s2_label1_embeddings', dpi=120)
    scatter(c_proj, labels)
    plt.savefig('data/c_embeddings', dpi=120)

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