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
    def __init__(self, train, valid, test, train_labels, valid_labels, test_labels, save_path, logging, num_epochs, log_interval,
                 warm_up, kl_start,
                 vae_params, lr_params):
        super(TrainerVAE, self).__init__()

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = save_path

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

        self.enc_optimizer = optim.Adam(self.vae.encoder.parameters(), lr=lr_params['enc_lr'])
        self.dec_optimizer = optim.Adam(self.vae.decoder.parameters(), lr=lr_params['dec_lr'])
        self.s_given_c_optimizer = optim.Adam(self.vae.s_given_c.parameters(), lr=lr_params['s_given_c_lr'])
        self.content_decoder_optimizer = optim.Adam(self.vae.content_decoder.parameters(),
                                                    lr=lr_params['content_decoder_lr'])
        self.style_classifier_optimizer = optim.Adam(self.vae.style_classifier.parameters(),
                                                     lr=lr_params['style_classifier_lr'])

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

        n_sample_c, batch_size_c, nc = c.size()
        n_sample_s, batch_size_s, ns = s.size()

        c = c.view(batch_size_c * n_sample_c, nc)
        s = s.view(batch_size_s * n_sample_s, ns)

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
        batch_size_y = train_labels.size()[0]
        assert (batch_size == batch_size_y)

        target = train_data[1:]

        content_logits = self.vae.content_decoder(train_data[:-1], c)
        content_logits = content_logits.view(-1, content_logits.size(2))
        content_rec_loss = F.cross_entropy(content_logits, target.view(-1), reduction="none")
        content_rec_loss = content_rec_loss.view(-1, batch_size).sum(0)

        style_logits = self.vae.style_classifier(s.squeeze())
        style_pred_loss = F.cross_entropy(style_logits, train_labels, reduction="none")
        style_pred_loss = style_pred_loss.view(-1, batch_size).sum(0)

        # total_dis_loss = content_rec_loss.mean() + style_pred_loss.mean() + self.vae.s_given_c.get_s_c_mi(s, c)
        return content_rec_loss.mean(), style_pred_loss.mean(), self.vae.s_given_c.get_s_c_mi(s, c)

    def train(self, epoch):
        self.vae.train()

        total_rec_loss = 0
        total_kl_loss = 0
        total_disent_loss = 0
        start_time = time.time()
        step = 0
        num_words = 0
        num_sents = 0

        for idx in np.random.permutation(range(self.nbatch)):
            batch_data = self.train_data[idx]
            batch_labels = torch.tensor(self.train_labels[idx])
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
            content_rec_loss, style_pred_loss, s_c_mi_loss = self.compute_disentangle_loss(batch_data, batch_labels)
            total_rec_loss += vae_rec_loss.sum().item()
            total_kl_loss += vae_kl.sum().item()
            total_disent_loss += content_rec_loss.item() + style_pred_loss.item() + s_c_mi_loss.item()
            loss = vae_loss + content_rec_loss + style_pred_loss + s_c_mi_loss

            loss.backward()

            nn.utils.clip_grad_norm_(self.vae.parameters(), 5.0)

            self.step_gradients()

            if content_rec_loss.item() <= 0:
                print("content_rec_loss ----------- Negative")
            if style_pred_loss.item() <= 0:
                print("style_pred_loss ------------ Negative")
            if s_c_mi_loss.item() <= 0:
                print("s_c_mi_loss     ------------ Negative")

            if step % self.log_interval == 0 and step > 0:
                cur_rec_loss = total_rec_loss / num_sents
                cur_kl_loss = total_kl_loss / num_sents
                cur_disent_loss = total_disent_loss / num_sents
                cur_vae_loss = cur_rec_loss + cur_kl_loss
                cur_total_loss = cur_vae_loss + cur_disent_loss
                elapsed = time.time() - start_time
                self.logging(
                    '| epoch {:2d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | loss {:3.2f} | vae_loss {:3.2f} | disent_loss {:3.2f} | '
                    'recon {:3.2f} | kl {:3.2f}'.format(
                        epoch, step, self.nbatch, elapsed * 1000 / self.log_interval, cur_total_loss, cur_vae_loss, cur_disent_loss,
                        cur_rec_loss, cur_kl_loss))
                total_rec_loss = 0
                total_kl_loss = 0
                total_disent_loss = 0
                num_sents = 0
                num_words = 0
                start_time = time.time()
            step += 1

    def evaluate(self):
        self.vae.eval()

        total_rec_loss = 0
        total_kl_loss = 0
        total_disent_loss = 0
        num_sents = 0
        num_words = 0

        with torch.no_grad():
            for idx, batch_data in enumerate(self.valid_data):
                sent_len, batch_size = batch_data.size()
                target = batch_data[1:]
                batch_labels = torch.tensor(self.valid_labels[idx])

                num_sents += batch_size
                num_words += (sent_len - 1) * batch_size

                vae_logits, vae_kl = self.vae.loss(batch_data)
                vae_logits = vae_logits.view(-1, vae_logits.size(2))
                vae_rec_loss = F.cross_entropy(vae_logits, target.view(-1), reduction="none")
                total_rec_loss += vae_rec_loss.sum().item()
                total_kl_loss += vae_kl.sum().item()

                content_rec_loss, style_pred_loss, s_c_mi_loss = self.compute_disentangle_loss(batch_data, batch_labels)
                total_disent_loss += content_rec_loss.item() + style_pred_loss.item() + s_c_mi_loss.item()

        cur_rec_loss = total_rec_loss / num_sents
        cur_kl_loss = total_kl_loss / num_sents
        cur_vae_loss = cur_rec_loss + cur_kl_loss
        cur_disent_loss = total_disent_loss / num_sents
        return cur_vae_loss, cur_rec_loss, cur_kl_loss, cur_disent_loss

    def fit(self):
        best_loss = 1e4
        decay_cnt = 0
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            self.train(epoch)
            val_losses = self.evaluate()
    
            vae_loss = val_losses[0] + val_losses[3]
    
            if vae_loss < best_loss:
                self.save(self.save_path)
                best_loss = vae_loss
            

            # if vae_loss > self.opt_dict["best_loss"]:
            #     self.opt_dict["not_improved"] += 1
            #     if self.opt_dict["not_improved"] >= 3 and epoch >= 15:
            #         self.opt_dict["not_improved"] = 0
            #         self.opt_dict["lr"] = self.opt_dict["lr"] * 0.5
            #         self.load(self.save_path)
            #         decay_cnt += 1
            #         self.enc_optimizer = optim.SGD(
            #             self.vae.encoder.parameters(), lr=self.opt_dict["lr"])
            #         self.dec_optimizer = optim.SGD(
            #             self.vae.decoder.parameters(), lr=self.opt_dict["lr"])
            # else:
            #     self.opt_dict["not_improved"] = 0
            #     self.opt_dict["best_loss"] = vae_loss
            # if decay_cnt == 5:
            #     break

            self.logging('-' * 75)
            # self.logging('| end of epoch {:2d} | time {:5.2f}s | '
            #              'kl_weight {:.2f} | vae_lr {:.2f} | loss {:3.2f}'.format(
            #                  epoch, (time.time() - epoch_start_time),
            #                  self.kl_weight, self.opt_dict["lr"], val_losses[0]))
            self.logging('| total_loss {:3.2f} | recon {:3.2f} | kl {:3.2f} | dient {:3.2f}'.format(
                val_losses[0], val_losses[1], val_losses[2], val_losses[3]))
            self.logging('-' * 75)
        
        return best_loss

    def save(self, path):
        self.logging("saving to %s" % path)
        model_path = os.path.join(path, "model.pt")
        torch.save(self.vae.state_dict(), model_path)

    def load(self, path):
        model_path = os.path.join(path, "model.pt")
        self.vae.load_state_dict(torch.load(model_path))