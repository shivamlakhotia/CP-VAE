import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from .utils import log_sum_exp
import numpy as np
from scipy.stats import ortho_group

class GaussianEncoderBase(nn.Module):
    def __init__(self):
        super(GaussianEncoderBase, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def sample(self, inputs, nsamples):
        mu, logvar = self.forward(inputs)
        z = self.reparameterize(mu, logvar, nsamples)
        return z, (mu, logvar)

    def encode(self, inputs, nsamples=1):
        mu, logvar = self.forward(inputs)
        z = self.reparameterize(mu, logvar, nsamples)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
        return z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(0).expand(nsamples, batch_size, nz)
        std_expd = std.unsqueeze(0).expand(nsamples, batch_size, nz)

        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)

    def sample_from_inference(self, x, nsamples=1):
        mu, logvar = self.forward(x)
        batch_size, nz = mu.size()
        return mu.unsqueeze(0).expand(nsamples, batch_size, nz)

    def eval_inference_dist(self, x, z, param=None):
        nz = z.size(2)
        if not param:
            mu, logvar = self.forward(x)
        else:
            mu, logvar = param

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()
        dev = z - mu

        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density.squeeze(0)

    def calc_mi(self, x):
        mu, logvar = self.forward(x)

        x_batch, nz = mu.size()

        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        z_samples = self.reparameterize(mu, logvar, 1)

        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        dev = z_samples - mu

        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        log_qz = log_sum_exp(log_density, dim=0) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()

class LSTMEncoder(GaussianEncoderBase):
    def __init__(self, ni, nh, nc, vocab_size, model_init, emb_init):
        super(LSTMEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, ni)

        self.lstm = nn.LSTM(input_size=ni,
                            hidden_size=nh,
                            num_layers=1,
                            bidirectional=True)
        self.linear = nn.Linear(nh, 2 * nc, bias=False)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs):
        if len(inputs.size()) > 2:
            word_embed = torch.matmul(inputs, self.embed.weight)
        else:
            word_embed = self.embed(inputs)

        outputs, (last_state, last_cell) = self.lstm(word_embed)
        seq_len, bsz, hidden_size = outputs.size()
        hidden_repr = outputs.view(seq_len, bsz, 2, -1).mean(2)
        hidden_repr = torch.max(hidden_repr, 0)[0]

        mean, logvar = self.linear(hidden_repr).chunk(2, -1)
        return mean, logvar