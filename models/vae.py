import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import torch.nn.functional as F

from .base_network import LSTMEncoder, StyleClassifier, ContentDecoder, SgivenC, LSTMDecoder
from .utils import uniform_initializer, value_initializer, gumbel_softmax


class VAE(nn.Module):
    def __init__(self, ni, nc, ns, n_attention_heads, enc_nh, dec_nh, dec_dropout_in, dec_dropout_out, num_styles,
                 vocab, device):
        super(VAE, self).__init__()
        model_init = uniform_initializer(0.01)
        enc_embed_init = uniform_initializer(0.1)
        dec_embed_init = uniform_initializer(0.1)
        self.encoder = LSTMEncoder(ni, enc_nh, nc, ns, n_attention_heads, len(vocab), model_init, enc_embed_init)
        self.decoder = LSTMDecoder(ni, nc, ns, dec_nh, vocab,
                                   model_init, dec_embed_init, device, dec_dropout_in, dec_dropout_out)
        self.s_given_c = SgivenC(nc, ns, model_init)
        self.content_decoder = ContentDecoder(ni, nc, dec_nh, vocab, model_init, dec_embed_init, device, dec_dropout_in,
                                              dec_dropout_out)
        self.style_classifier = StyleClassifier(ns, num_styles)

    def cuda(self):
        self.encoder.cuda()
        self.decoder.cuda()
        self.s_given_c.cuda()
        self.content_decoder.cuda()
        self.style_classifier.cuda()

    def encode(self, x, nsamples=1):
        return self.encoder.encode(x, nsamples)

    def decode(self, x, c, s):
        return self.decoder(x, c, s)

    def loss(self, x, nsamples=1):
        c, s, KL = self.encode(x, nsamples)
        outputs = self.decode(x[:-1], c, s)
        return outputs, KL

    def calc_mi_q(self, x):
        assert False
        return self.encoder.calc_mi(x)


class TrainerVAE:
    def __init__(self, train, valid, test, train_labels, valid_labels, test_labels, logging, num_epochs, log_interval, warm_up, kl_start,
                 vae_params):
        super(TrainerVAE, self).__init__()

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_data = train
        self.valid_data = valid
        self.test_data = test
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels

        self.logging = logging
        self.log_interval = log_interval
        self.warm_up = warm_up
        self.kl_weight = kl_start
        self.num_epochs = num_epochs
        self.opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

        self.vae = VAE(**vae_params)
        if self.use_cuda:
            self.vae.cuda()

        self.enc_optimizer = optim.SGD(self.vae.encoder.parameters(), lr=vae_params['enc_lr'])
        self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(), lr=vae_params['dec_lr'])
        self.s_given_c_optimizer = optim.SGD(self.vae.s_given_c.parameters(), lr=vae_params['s_given_c_lr'])
        self.content_decoder_optimizer = optim.SGD(self.vae.content_decoder.parameters(),
                                                   lr=vae_params['content_decoder_lr'])
        self.style_classifier_optimizer = optim.SGD(self.vae.style_classifier.parameters(),
                                                    lr=vae_params['style_classifier_lr'])

        self.nbatch = len(self.train_data)
        self.anneal_rate = (1.0 - kl_start) / (warm_up * self.nbatch)

    def reset_gradients(self):
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()
        # self.s_given_c_optimizer.zero_grad()
        self.content_decoder_optimizer.zero_grad()
        self.style_classifier_optimizer.zero_grad()

    def step_gradients(self):
        self.enc_optimizer.step()
        self.dec_optimizer.step()
        self.content_decoder_optimizer.step()
        self.style_classifier_optimizer.step()

    def train_s_given_c(self, batch_data):
        c, s, _ = self.vae.encoder.encode(batch_data)
        assert (c.size(0) == 1)

        batch_size = c.size(1)
        c = c.view(batch_size, -1)
        s = s.view(batch_size, -1)

        self.s_given_c_optimizer.zero_grad()

        c = c.detach()
        s = s.detach()

        loss = self.vae.s_given_c.get_log_prob_total(s, c)
        loss.backward()
        self.s_given_c_optimizer.step()

    def compute_disentangle_loss(self, train_data, train_labels):
        c_, s_, _ = self.vae.encoder.encode(train_data)

        c = c_.detach()  # TODO: Is this needed!??
        s = s_.detach()

        sent_len, batch_size = train_data.size()
        batch_size_y = train_labels.size()
        assert(batch_size == batch_size_y)

        target = train_data[1:]

        content_logits = self.vae.content_decoder(train_data[:-1], c)
        content_logits = content_logits.view(-1, content_logits.size(2))
        content_rec_loss = F.cross_entropy(content_logits, target.view(-1), reduction="none")
        content_rec_loss = content_rec_loss.view(-1, batch_size).sum(0)

        style_logits = self.vae.style_classifier(s)
        style_pred_loss = F.cross_entropy(style_logits, train_labels, reduction="none")
        style_pred_loss = style_pred_loss.view(-1, batch_size).sum(0)

        total_dis_loss = content_rec_loss.mean() + style_pred_loss.mean() + self.vae.s_given_c.get_s_c_mi(s, c)
        return total_dis_loss

    def train(self, epoch):
        self.vae.train()

        total_rec_loss = 0
        total_kl_loss = 0
        start_time = time.time()
        step = 0
        num_words = 0
        num_sents = 0

        for idx in np.random.permutation(range(self.nbatch)):
            batch_data = self.train_data[idx]
            batch_labels = self.train_labels[idx]
            sent_len, batch_size = batch_data.size()

            target = batch_data[1:]
            num_words += (sent_len - 1) * batch_size
            num_sents += batch_size
            self.kl_weight = min(1.0, self.kl_weight + self.anneal_rate)
            self.kl_weight = 0.35

            self.reset_gradients()
            self.train_s_given_c(batch_data)

            vae_logits, vae_kl = self.vae.loss(batch_data)
            vae_logits = vae_logits.view(-1, vae_logits.size(2))
            vae_rec_loss = F.cross_entropy(vae_logits, target.view(-1), reduction="none")
            vae_rec_loss = vae_rec_loss.view(-1, batch_size).sum(0)
            vae_loss = vae_rec_loss + self.kl_weight * vae_kl
            vae_loss = vae_loss.mean()
            total_rec_loss += vae_rec_loss.sum().item()
            total_kl_loss += vae_kl.sum().item()
            loss = vae_loss + self.compute_disentangle_loss(batch_data, batch_labels)

            loss.backward()

            nn.utils.clip_grad_norm_(self.vae.parameters(), 5.0)

            self.step_gradients()

            if step % self.log_interval == 0 and step > 0:
                cur_rec_loss = total_rec_loss / num_sents
                cur_kl_loss = total_kl_loss / num_sents
                cur_vae_loss = cur_rec_loss + cur_kl_loss
                elapsed = time.time() - start_time
                self.logging(
                    '| epoch {:2d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | loss {:3.2f} | '
                    'recon {:3.2f} | kl {:3.2f}'.format(
                        epoch, step, self.nbatch, elapsed * 1000 / self.log_interval, cur_vae_loss,
                        cur_rec_loss, cur_kl_loss))
                total_rec_loss = 0
                total_kl_loss = 0
                num_sents = 0
                num_words = 0
                start_time = time.time()
            step += 1

    # def fit(self):
    #     best_loss = 1e4
    #     decay_cnt = 0
    #     for epoch in range(1, self.num_epochs + 1):
    #         epoch_start_time = time.time()
    #         self.train(epoch)
    #         val_loss = self.evaluate(self.valid_data)
    #
    #         vae_loss = val_loss[1]
    #
    #         if self.aggressive:
    #             cur_mi = val_loss[-1]
    #             self.logging("pre mi: %.4f, cur mi:%.4f" % (self.pre_mi, cur_mi))
    #             if cur_mi < self.pre_mi:
    #                 self.aggressive = False
    #                 self.logging("STOP BURNING")
    #
    #             self.pre_mi = cur_mi
    #
    #         if vae_loss < best_loss:
    #             self.save(self.save_path)
    #             best_loss = vae_loss