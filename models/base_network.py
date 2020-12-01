import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from models.utils import log_sum_exp
import numpy as np
from scipy.stats import ortho_group
from torch.distributions.multivariate_normal import MultivariateNormal

class GaussianEncoderBase(nn.Module):
    def __init__(self):
        super(GaussianEncoderBase, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def sample(self, inputs, nsamples):
        mu_c, logvar_c, mu_s, logvar_s = self.forward(inputs)
        c = self.reparameterize(mu_c, logvar_c, nsamples)
        s = self.reparameterize(mu_s, logvar_s, nsamples)
        return c, s, (mu_c, logvar_c), (mu_s, logvar_s)

    def encode(self, inputs, nsamples=1):
        mu_c, logvar_c, mu_s, logvar_s = self.forward(inputs)
        c = self.reparameterize(mu_c, logvar_c, nsamples)
        s = self.reparameterize(mu_s, logvar_s, nsamples)

        KL = 0.5 * (mu_c.pow(2) + logvar_c.exp() - logvar_c - 1).sum(1) + 0.5 * (
                    mu_s.pow(2) + logvar_s.exp() - logvar_s - 1).sum(1)
        return c, s, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(0).expand(nsamples, batch_size, nz)
        std_expd = std.unsqueeze(0).expand(nsamples, batch_size, nz)

        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)

    def sample_from_inference(self, x, nsamples=1):
        assert False
        mu, logvar = self.forward(x)
        batch_size, nz = mu.size()
        return mu.unsqueeze(0).expand(nsamples, batch_size, nz)

    def eval_inference_dist(self, x, z, param=None):
        assert False
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
    def __init__(self, ni, nh, nc, ns, attention_heads, vocab_size, model_init, emb_init):
        super(LSTMEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, ni)

        self.lstm = nn.LSTM(input_size=ni,
                            hidden_size=nh,
                            num_layers=1,
                            bidirectional=True)
        self.attention_layer = nn.MultiheadAttention(nh, attention_heads)
        self.linear_c = nn.Linear(nh, 2 * nc, bias=False)
        self.linear_s = nn.Linear(nh, 2 * ns, bias=False)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs):
        if len(inputs.size()) > 2:
            assert False
            word_embed = torch.matmul(inputs, self.embed.weight)
        else:
            word_embed = self.embed(inputs)

        outputs, (last_state, last_cell) = self.lstm(word_embed)
        outputs, _ = self.attention_layer(outputs, outputs, outputs)
        seq_len, bsz, hidden_size = outputs.size()
        hidden_repr = outputs.view(seq_len, bsz, 2, -1).mean(2)
        hidden_repr = torch.max(hidden_repr, 0)[0]

        mean_c, logvar_c = self.linear_c(hidden_repr).chunk(2, -1)
        mean_s, logvar_s = self.linear_s(hidden_repr).chunk(2, -1)
        return mean_c, logvar_c, mean_s, logvar_s


class StyleClassifier(nn.Module):
    def __init__(self, ns, num_labels):
        super(StyleClassifier, self).__init__()
        self.linear = nn.Linear(ns, 2**num_labels, bias=False)

    def forward(self, inputs):
        output = self.linear(inputs)
        # output = nn.Softmax(output)
        return output


class ContentDecoder(nn.Module):
    def __init__(self, ni, nc, nh, vocab, model_init, emb_init, device, dropout_in=0, dropout_out=0):
        super(ContentDecoder, self).__init__()
        self.vocab = vocab
        self.device = device

        self.embed = nn.Embedding(len(vocab), ni, padding_idx=-1)

        self.lstm = nn.LSTM(input_size=nc + ni,
                            hidden_size=nh,
                            num_layers=1,
                            bidirectional=False)

        self.pred_linear = nn.Linear(nh, len(vocab), bias=False)
        self.trans_linear = nn.Linear(nc, nh, bias=False)

        self.dropout_in = nn.Dropout(dropout_in)
        self.dropout_out = nn.Dropout(dropout_out)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs, c):
        n_sample_c, batch_size_c, nc = c.size()

        seq_len = inputs.size(0)

        word_embed = self.embed(inputs)
        word_embed = self.dropout_in(word_embed)

        if n_sample_c == 1:
            c_ = c.expand(seq_len, batch_size_c, nc)
        else:
            raise NotImplementedError

        word_embed = torch.cat((word_embed, c_), -1)

        c = c.view(batch_size_c * n_sample_c, nc)

        c_init = self.trans_linear(c).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)
        output_logits = self.pred_linear(output)

        return output_logits.view(-1, batch_size_c, len(self.vocab))

    def decode(self, c, greedy=True):
        n_sample_c, batch_size_c, nc = c.size()

        assert (n_sample_c == 1)

        c = c.view(batch_size_c * n_sample_c, nc)

        batch_size = c.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        c_init = self.trans_linear(c).unsqueeze(0)
        h_init = torch.tanh(c_init)
        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long,
                                     device=self.device).unsqueeze(0)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long,
                                  device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:
            word_embed = self.embed(decoder_input)

            word_embed = torch.cat((word_embed, c.unsqueeze(0)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(0)

            if greedy:
                select_index = torch.argmax(output_logits, dim=1)
            else:
                sample_prob = F.softmax(output_logits, dim=1)
                select_index = torch.multinomial(sample_prob, num_samples=1).squeeze(1)

            decoder_input = select_index.unsqueeze(0)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.id2word(select_index[i].item()))

            mask = torch.mul((select_index != end_symbol), mask)

        return decoded_batch


class SgivenC(nn.Module):
    def __init__(self, nc, ns, model_init):
        super(SgivenC, self).__init__()
        self.linear1 = nn.Linear(nc, 128, bias=False)
        self.linear2 = nn.Linear(128, 2*ns, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters(model_init)

    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, c):
        output = self.relu(self.linear1(c))
        mean_s, logvar_s = self.linear2(output).chunk(2, -1)

        return mean_s, logvar_s

    def get_log_prob_single(self, s, mean_s, logvar_s):
        cov_s = torch.diag(logvar_s.exp())
        m = MultivariateNormal(mean_s, cov_s)

        return m.log_prob(s)

    def get_log_prob_total(self, s, c):
        mean_s, logvar_s = self.forward(c)
        batch_size = mean_s.size(0)
        loss = 0
        for i in range(batch_size):
            loss += self.get_log_prob_single(s[i], mean_s[i], logvar_s[i])

        return loss/batch_size

    def get_s_c_mi(self, s, c):
        mean_s, logvar_s = self.forward(c)
        batch_size = mean_s.size(0)
        loss = 0
        for i in range(batch_size):
            j = torch.randint(low=0, high=batch_size+1, size=(1,))
            loss += self.get_log_prob_single(s[i], mean_s[i], logvar_s[i]) - self.get_log_prob_single(s[i], mean_s[j], logvar_s[j])

        return loss/batch_size

class LSTMDecoder(nn.Module):
    def __init__(self, ni, nc, ns, nh, vocab, model_init, emb_init, device, dropout_in=0, dropout_out=0):
        super(LSTMDecoder, self).__init__()
        self.vocab = vocab
        self.device = device

        self.lstm = nn.LSTM(input_size=ni + ns + nc,
                            hidden_size=nh,
                            num_layers=2,
                            bidirectional=False)

        self.pred_linear = nn.Linear(nh, len(vocab), bias=False)
        self.trans_linear = nn.Linear(ns + nc, nh, bias=False)

        self.embed = nn.Embedding(len(vocab), ni, padding_idx=-1)

        self.dropout_in = nn.Dropout(dropout_in)
        self.dropout_out = nn.Dropout(dropout_out)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs, c, s):
        n_sample_c, batch_size_c, nc = c.size()
        n_sample_s, batch_size_s, ns = s.size()

        assert (n_sample_c == n_sample_s)
        assert (batch_size_c == batch_size_s)

        seq_len = inputs.size(0)

        word_embed = self.embed(inputs)
        word_embed = self.dropout_in(word_embed)

        if n_sample_c == 1:
            c_ = c.expand(seq_len, batch_size_c, nc)
            s_ = s.expand(seq_len, batch_size_s, ns)
        else:
            raise NotImplementedError

        word_embed = torch.cat((word_embed, c_, s_), -1)

        c = c.view(batch_size_c * n_sample_c, nc)
        s = s.view(batch_size_s * n_sample_s, ns)

        concat_c_s = torch.cat((c, s), 1)

        c_init = self.trans_linear(concat_c_s).unsqueeze(0).repeat(2, 1, 1)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)
        output_logits = self.pred_linear(output)

        return output_logits.view(-1, batch_size_c, len(self.vocab))

    def decode(self, c, s, greedy=True):
        n_sample_c, batch_size_c, nc = c.size()
        n_sample_s, batch_size_s, ns = s.size()

        assert (n_sample_c == 1 and n_sample_s == 1)

        c = c.view(batch_size_c * n_sample_c, nc)
        s = s.view(batch_size_s * n_sample_s, ns)

        z = torch.cat((c, s), 1)
        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long,
                                     device=self.device).unsqueeze(0)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long,
                                  device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:
            word_embed = self.embed(decoder_input)

            word_embed = torch.cat((word_embed, z.unsqueeze(0)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(0)

            if greedy:
                select_index = torch.argmax(output_logits, dim=1)
            else:
                sample_prob = F.softmax(output_logits, dim=1)
                select_index = torch.multinomial(sample_prob, num_samples=1).squeeze(1)

            decoder_input = select_index.unsqueeze(0)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.id2word(select_index[i].item()))

            mask = torch.mul((select_index != end_symbol), mask)

        return decoded_batch
