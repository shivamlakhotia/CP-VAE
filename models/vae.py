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
        self.encoder = LSTMEncoder(ni, enc_nh, nc, ns, n_attention_heads, vocab, device, model_init, enc_embed_init)
        self.decoder = LSTMDecoder(ni, nc, ns, dec_nh, vocab,
                                   model_init, dec_embed_init, device, dec_dropout_in, dec_dropout_out)
        self.s1_given_c = SgivenC(nc, ns, model_init)
        self.s2_given_c = SgivenC(nc, ns, model_init)
        self.s1_given_s2 = SgivenC(ns, ns, model_init)
        self.content_decoder = ContentDecoder(ni, nc, dec_nh, vocab, model_init, dec_embed_init, device, dec_dropout_in,
                                              dec_dropout_out)
        self.style_classifier1 = StyleClassifier(ns, num_styles)
        self.style_classifier2 = StyleClassifier(ns, num_styles)

    def cuda(self):
        self.encoder.cuda()
        self.decoder.cuda()
        self.s1_given_c.cuda()
        self.s2_given_c.cuda()
        self.s1_given_s2.cuda()
        self.content_decoder.cuda()
        self.style_classifier1.cuda()
        self.style_classifier2.cuda()

    def encode(self, x, nsamples=1):
        return self.encoder.encode(x, nsamples)

    def decode(self, x, c, s1, s2):
        return self.decoder(x, c, s1, s2)

    def loss(self, x, nsamples=1):
        c, s1, s2, KL = self.encode(x, nsamples)
        outputs = self.decode(x[:-1], c, s1, s2)
        return outputs, KL

    def calc_mi_q(self, x):
        assert False
        return self.encoder.calc_mi(x)
    
    def set_s_given_c_state(self, freeze):
        for p in self.s1_given_c.parameters():
            p.requires_grad = freeze
        for p in self.s2_given_c.parameters():
            p.requires_grad = freeze
        for p in self.s1_given_s2.parameters():
            p.requires_grad = freeze


class TrainerVAE:
    def __init__(self, train, valid, test, train_labels1, train_labels2, valid_labels1, valid_labels2, test_labels1, test_labels2, save_path, logging, num_epochs, log_interval,
                 warm_up, kl_start, vae_params, lr_params, dist_train):
        super(TrainerVAE, self).__init__()

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = save_path

        self.train_data = train
        self.valid_data = valid
        self.test_data = test
        self.train_labels1 = train_labels1
        self.valid_labels1 = valid_labels1
        self.test_labels1 = test_labels1
        self.train_labels2 = train_labels2
        self.valid_labels2 = valid_labels2
        self.test_labels2 = test_labels2

        self.dist_train = dist_train
        self.logging = logging
        self.log_interval = log_interval
        self.warm_up = warm_up
        self.kl_weight = kl_start
        self.num_epochs = num_epochs
        self.opt_dict = {"not_improved": 0, "enc_lr": lr_params['enc_lr'], "dec_lr": lr_params['dec_lr'], "s_given_c_lr": lr_params['s_given_c_lr'], 
            "content_decoder_lr": lr_params['content_decoder_lr'], "style_classifier_lr": lr_params['style_classifier_lr'], "best_loss": 1e4}

        self.vae = VAE(**vae_params)
        if self.use_cuda:
            self.vae.cuda()

        self.enc_optimizer = optim.Adam(self.vae.encoder.parameters(), lr=self.opt_dict['enc_lr'])
        # self.dec_optimizer = optim.Adam(self.vae.decoder.parameters(), lr=self.opt_dict['dec_lr'])
        self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(), lr=self.opt_dict['dec_lr'])
        self.s1_given_c_optimizer = optim.Adam(self.vae.s1_given_c.parameters(), lr=self.opt_dict['s_given_c_lr'])
        self.s2_given_c_optimizer = optim.Adam(self.vae.s2_given_c.parameters(), lr=self.opt_dict['s_given_c_lr'])
        self.s1_given_s2_optimizer = optim.Adam(self.vae.s1_given_s2.parameters(), lr=self.opt_dict['s_given_c_lr'])
        self.content_decoder_optimizer = optim.SGD(self.vae.content_decoder.parameters(),
                                                    lr=self.opt_dict['content_decoder_lr'])
        self.style_classifier1_optimizer = optim.Adam(self.vae.style_classifier1.parameters(), lr=self.opt_dict['style_classifier_lr'])
        self.style_classifier2_optimizer = optim.Adam(self.vae.style_classifier2.parameters(), lr=self.opt_dict['style_classifier_lr'])

        self.nbatch = len(self.train_data)
        self.anneal_rate = (1.0 - kl_start) / (warm_up * self.nbatch)
        self.beta_weight = 0

    def reset_gradients(self):
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()
        # self.s_given_c_optimizer.zero_grad()
        self.content_decoder_optimizer.zero_grad()
        self.style_classifier1_optimizer.zero_grad()
        self.style_classifier2_optimizer.zero_grad()

    def step_gradients(self):
        self.enc_optimizer.step()
        self.dec_optimizer.step()
        self.content_decoder_optimizer.step()
        self.style_classifier1_optimizer.step()
        self.style_classifier2_optimizer.step()

    def train_s_given_c(self, batch_data):
        c, s1, s2, _ = self.vae.encoder.encode(batch_data)
        assert (c.size(0) == 1)

        n_sample_c, batch_size_c, nc = c.size()
        n_sample_s1, batch_size_s1, ns1 = s1.size()
        n_sample_s2, batch_size_s2, ns2 = s2.size()

        c = c.view(batch_size_c * n_sample_c, nc)
        s1 = s1.view(batch_size_s1 * n_sample_s1, ns1)
        s2 = s2.view(batch_size_s2 * n_sample_s2, ns2)

        self.vae.set_s_given_c_state(True)
        self.s1_given_c_optimizer.zero_grad()
        self.s2_given_c_optimizer.zero_grad()
        self.s1_given_s2_optimizer.zero_grad()

        c = c.detach()
        s1 = s1.detach()
        s2 = s2.detach()

        loss = -(self.vae.s1_given_c.get_log_prob_total(s1, c) + self.vae.s2_given_c.get_log_prob_total(s2, c) + self.vae.s1_given_s2.get_log_prob_total(s1, s2))
        loss.backward()
        self.s1_given_c_optimizer.step()
        self.s2_given_c_optimizer.step()
        self.s1_given_s2_optimizer.step()

        return loss.item()

    def train_s_given_c_distribution(self, batch_data):
        mu_c, logvar_c, mu_s1, logvar_s1, mu_s2, logvar_s2 = self.vae.encoder(batch_data)

        c = self.vae.encoder.reparameterize(mu_c, logvar_c, 1)
        s2 = self.vae.encoder.reparameterize(mu_s2, logvar_s2, 1)
        assert (c.size(0) == 1)
        
        n_sample_c, batch_size_c, nc = c.size()
        n_sample_s2, batch_size_s2, ns2 = s2.size()

        c = c.view(batch_size_c * n_sample_c, nc)
        s2 = s2.view(batch_size_s2 * n_sample_s2, ns2)

        c = c.detach()
        s2 = s2.detach()
        mu_s1 = mu_s1.detach()
        logvar_s1 = logvar_s1.detach()
        mu_s2 = mu_s2.detach()
        logvar_s2 = logvar_s2.detach()

        self.vae.set_s_given_c_state(True)
        self.s1_given_c_optimizer.zero_grad()
        self.s2_given_c_optimizer.zero_grad()
        self.s1_given_s2_optimizer.zero_grad()

        loss = self.vae.s1_given_c.get_log_prob_total_distribution(mu_s1, logvar_s1, c) + self.vae.s2_given_c.get_log_prob_total_distribution(mu_s2, logvar_s2, c) + \
            self.vae.s1_given_s2.get_log_prob_total_distribution(mu_s1, logvar_s1, s2)
        loss.backward()
        self.s1_given_c_optimizer.step()
        self.s2_given_c_optimizer.step()
        self.s1_given_s2_optimizer.step()

        return loss.item()

    def compute_disentangle_loss(self, train_data, train_labels1, train_labels2):
        c, s1, s2, _ = self.vae.encoder.encode(train_data)

        # c = c_.detach()  # TODO: Is this needed!??
        # s = s_.detach()

        self.vae.set_s_given_c_state(False)

        sent_len, batch_size = train_data.size()
        batch_size_y1 = train_labels1.size()[0]
        batch_size_y2 = train_labels2.size()[0]
        assert (batch_size == batch_size_y1 and batch_size == batch_size_y2)

        target = train_data[1:]

        content_logits = self.vae.content_decoder(train_data[:-1], c)
        content_logits = content_logits.view(-1, content_logits.size(2))
        content_rec_loss = F.cross_entropy(content_logits, target.view(-1), reduction="none")
        content_rec_loss = content_rec_loss.view(-1, batch_size).sum(0)

        style_logits1 = self.vae.style_classifier1(s1.squeeze())
        style_pred_loss1 = F.cross_entropy(style_logits1, train_labels1, reduction="none")
        style_pred_loss1 = style_pred_loss1.view(-1, batch_size).sum(0)
        style_logits2 = self.vae.style_classifier2(s2.squeeze())
        style_pred_loss2 = F.cross_entropy(style_logits2, train_labels2, reduction="none")
        style_pred_loss2 = style_pred_loss2.view(-1, batch_size).sum(0)

        loss1 = content_rec_loss.mean()
        loss2 = style_pred_loss1.mean()
        loss3 = style_pred_loss2.mean()
        loss4 = self.vae.s1_given_c.get_s_c_mi(s1, c)
        loss5 = self.vae.s2_given_c.get_s_c_mi(s2, c)
        loss6 = self.vae.s1_given_s2.get_s_c_mi(s1, s2)
        total_dis_loss = loss1 + 10*loss2 + 10*loss3 + loss4 + loss5 + loss6
        return total_dis_loss, loss1.item(), loss2.item()+loss3.item(), loss4.item()+loss5.item()+loss6.item()

    def compute_disentangle_loss_distribution(self, train_data, train_labels1, train_labels2):
        mu_c, logvar_c, mu_s1, logvar_s1, mu_s2, logvar_s2 = self.vae.encoder(train_data)

        # c = c_.detach()  # TODO: Is this needed!??
        # s = s_.detach()

        c = self.vae.encoder.reparameterize(mu_c, logvar_c, 1)
        s1 = self.vae.encoder.reparameterize(mu_s1, logvar_s1, 1)
        s2 = self.vae.encoder.reparameterize(mu_s2, logvar_s2, 1)
        assert (c.size(0) == 1)

        self.vae.set_s_given_c_state(False)

        sent_len, batch_size = train_data.size()
        batch_size_y1 = train_labels1.size()[0]
        batch_size_y2 = train_labels2.size()[0]
        assert (batch_size == batch_size_y1 and batch_size == batch_size_y2)

        target = train_data[1:]

        content_logits = self.vae.content_decoder(train_data[:-1], c)
        content_logits = content_logits.view(-1, content_logits.size(2))
        content_rec_loss = F.cross_entropy(content_logits, target.view(-1), reduction="none")
        content_rec_loss = content_rec_loss.view(-1, batch_size).sum(0)

        style_logits1 = self.vae.style_classifier1(s1.squeeze())
        style_pred_loss1 = F.cross_entropy(style_logits1, train_labels1, reduction="none")
        style_pred_loss1 = style_pred_loss1.view(-1, batch_size).sum(0)
        style_logits2 = self.vae.style_classifier2(s2.squeeze())
        style_pred_loss2 = F.cross_entropy(style_logits2, train_labels2, reduction="none")
        style_pred_loss2 = style_pred_loss2.view(-1, batch_size).sum(0)

        loss1 = content_rec_loss.mean()
        loss2 = style_pred_loss1.mean()
        loss3 = style_pred_loss2.mean()
        loss4 = self.vae.s1_given_c.get_s_c_mi_distribution(mu_s1, logvar_s1, c)
        loss5 = self.vae.s2_given_c.get_s_c_mi_distribution(mu_s2, logvar_s2, c)
        loss6 = self.vae.s1_given_s2.get_s_c_mi_distribution(mu_s1, logvar_s1, s2)
        total_dis_loss = loss1 + 10*loss2 + 10*loss3 + loss4 + loss5 + loss6
        return total_dis_loss, loss1.item(), loss2.item()+loss3.item(), loss4.item()+loss5.item()+loss6.item()

    def train(self, epoch):
        self.vae.train()

        total_rec_loss = 0
        total_kl_loss = 0
        total_disent_loss = 0
        total_content_rec_loss = 0
        total_style_pred_loss = 0
        total_s_c_disent_loss = 0
        total_s_given_c_loss = 0
        start_time = time.time()
        step = 0
        num_words = 0
        num_sents = 0

        self.beta_weight += 0.05
        self.beta_weight = min(1.0, self.beta_weight)

        for idx in np.random.permutation(range(self.nbatch)):
            # self.vae.train()
            batch_data = self.train_data[idx]
            batch_labels1 = self.train_labels1[idx]
            batch_labels2 = self.train_labels2[idx]
            sent_len, batch_size = batch_data.size()

            target = batch_data[1:]
            num_words += (sent_len - 1) * batch_size
            num_sents += batch_size
            self.kl_weight = min(1.0, self.kl_weight + 0.3*self.anneal_rate)
            # self.kl_weight = 0.35

            self.reset_gradients()
            if self.dist_train:
                total_s_given_c_loss += self.train_s_given_c_distribution(batch_data)
            else:
                total_s_given_c_loss += self.train_s_given_c(batch_data)
            

            vae_logits, vae_kl = self.vae.loss(batch_data)
            vae_logits = vae_logits.view(-1, vae_logits.size(2))
            vae_rec_loss = F.cross_entropy(vae_logits, target.view(-1), reduction="none")
            vae_rec_loss = vae_rec_loss.view(-1, batch_size).sum(0)
            vae_loss = vae_rec_loss + self.kl_weight * vae_kl
            vae_loss = vae_loss.mean()

            if self.dist_train:
                disent_loss, content_rec_loss, style_pred_loss, s_c_disent_loss = self.compute_disentangle_loss_distribution(batch_data, batch_labels1, batch_labels2)
            else:
                disent_loss, content_rec_loss, style_pred_loss, s_c_disent_loss = self.compute_disentangle_loss(batch_data, batch_labels1, batch_labels2)
            total_rec_loss += vae_rec_loss.sum().item()
            total_kl_loss += vae_kl.sum().item()
            total_disent_loss += disent_loss.item()
            total_content_rec_loss += content_rec_loss
            total_style_pred_loss += style_pred_loss
            total_s_c_disent_loss += s_c_disent_loss

            loss = vae_loss + self.beta_weight * disent_loss

            loss.backward()

            nn.utils.clip_grad_norm_(self.vae.parameters(), 5.0)

            self.step_gradients()

            if step % self.log_interval == 0 and step > 0:
                cur_rec_loss = total_rec_loss / num_sents
                cur_kl_loss = total_kl_loss / num_sents
                cur_disent_loss = total_disent_loss / self.log_interval
                cur_content_rec_loss = total_content_rec_loss / self.log_interval
                cur_style_pred_loss = total_style_pred_loss / self.log_interval
                cur_s_c_disent_loss = total_s_c_disent_loss / self.log_interval
                cur_s_given_c_loss = total_s_given_c_loss / self.log_interval
                cur_vae_loss = cur_rec_loss + cur_kl_loss
                cur_total_loss = cur_vae_loss + cur_disent_loss
                elapsed = time.time() - start_time
                self.logging(
                    '| epoch {:2d} | {:5d}/{:5d} batches | {:5.4f} sec / batch | loss {:3.4f} | vae_loss {:3.4f} | disent_loss {:3.4f} | '
                    'content_rec_loss {:3.4f} | style_pred_loss {:3.4f} | s_c_disent_loss {:3.4f} | '
                    'recon {:3.4f} | kl {:3.4f} | s_given_c_loss {:3.4f}'.format(
                        epoch, step, self.nbatch, elapsed / self.log_interval, cur_total_loss, cur_vae_loss, cur_disent_loss, 
                        cur_content_rec_loss, cur_style_pred_loss, cur_s_c_disent_loss,
                        cur_rec_loss, cur_kl_loss, cur_s_given_c_loss))
                total_rec_loss = 0
                total_kl_loss = 0
                total_disent_loss = 0
                total_content_rec_loss = 0
                total_style_pred_loss = 0
                total_s_c_disent_loss = 0
                total_s_given_c_loss = 0
                num_sents = 0
                num_words = 0
                start_time = time.time()

                # val_losses = self.evaluate()
                # self.logging('| Validation DEBUG!!: total_loss {:3.2f} | recon {:3.2f} | kl {:3.2f} | dient {:3.2f}'.format(
                # val_losses[0]+val_losses[3], val_losses[1], val_losses[2], val_losses[3]))
            step += 1

    def evaluate(self):
        self.vae.eval()

        total_rec_loss = 0
        total_kl_loss = 0
        total_disent_loss = 0
        total_content_rec_loss = 0
        total_style_pred_loss = 0
        total_s_c_disent_loss = 0
        num_sents = 0
        num_words = 0
        num_batches = 0

        with torch.no_grad():
            for idx, batch_data in enumerate(self.valid_data):
                sent_len, batch_size = batch_data.size()
                target = batch_data[1:]
                batch_labels1 = self.valid_labels1[idx]
                batch_labels2 = self.valid_labels2[idx]

                num_sents += batch_size
                num_batches += 1
                num_words += (sent_len - 1) * batch_size

                vae_logits, vae_kl = self.vae.loss(batch_data)
                vae_logits = vae_logits.view(-1, vae_logits.size(2))
                vae_rec_loss = F.cross_entropy(vae_logits, target.view(-1), reduction="none")
                total_rec_loss += vae_rec_loss.sum().item()
                total_kl_loss += vae_kl.sum().item()

                if self.dist_train:
                    disent_loss, content_rec_loss, style_pred_loss, s_c_disent_loss = self.compute_disentangle_loss_distribution(batch_data, batch_labels1, batch_labels2)
                else:
                    disent_loss, content_rec_loss, style_pred_loss, s_c_disent_loss = self.compute_disentangle_loss(batch_data, batch_labels1, batch_labels2)
                total_disent_loss += disent_loss.item()
                total_content_rec_loss += content_rec_loss
                total_style_pred_loss += style_pred_loss
                total_s_c_disent_loss += s_c_disent_loss

        cur_rec_loss = total_rec_loss / num_sents
        cur_kl_loss = total_kl_loss / num_sents
        cur_vae_loss = cur_rec_loss + cur_kl_loss
        cur_disent_loss = total_disent_loss / num_batches
        cur_content_rec_loss = total_content_rec_loss / num_batches
        cur_style_pred_loss = total_style_pred_loss / num_batches
        cur_s_c_disent_loss = total_s_c_disent_loss / num_batches
        return cur_vae_loss, cur_rec_loss, cur_kl_loss, cur_disent_loss, cur_content_rec_loss, cur_style_pred_loss, cur_s_c_disent_loss

    def fit(self):
        best_loss = 1e4
        best_style_pred_loss = 1e4
        decay_cnt = 0
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            self.train(epoch)
            val_losses = self.evaluate()
    
            total_loss = val_losses[0] + val_losses[3]
    
            # self.save(self.save_path)
            if total_loss < best_loss:
                self.save(self.save_path)
                best_loss = total_loss
            
            # if val_losses[5] < best_style_pred_loss:
            #     self.save(os.path.join(self.save_path, 'style_save'))
            #     best_style_pred_loss = val_losses[5]

            if total_loss > self.opt_dict["best_loss"]:
                self.opt_dict["not_improved"] += 1
                if self.opt_dict["not_improved"] >= 3 and epoch >= 10:
                    self.opt_dict["not_improved"] = 0
                    self.opt_dict["enc_lr"] = self.opt_dict["enc_lr"] * 0.5
                    self.opt_dict["dec_lr"] = self.opt_dict["dec_lr"] * 0.5
                    self.opt_dict["s_given_c_lr"] = self.opt_dict["s_given_c_lr"] * 0.5
                    self.opt_dict["content_decoder_lr"] = self.opt_dict["content_decoder_lr"] * 0.5
                    self.opt_dict["style_classifier_lr"] = self.opt_dict["style_classifier_lr"] * 0.5
                    self.load(self.save_path)
                    decay_cnt += 1

                    self.enc_optimizer = optim.Adam(self.vae.encoder.parameters(), lr=self.opt_dict['enc_lr'])
                    self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(), lr=self.opt_dict['dec_lr'])
                    self.s1_given_c_optimizer = optim.Adam(self.vae.s1_given_c.parameters(), lr=self.opt_dict['s_given_c_lr'])
                    self.s2_given_c_optimizer = optim.Adam(self.vae.s2_given_c.parameters(), lr=self.opt_dict['s_given_c_lr'])
                    self.s1_given_s2_optimizer = optim.Adam(self.vae.s1_given_s2.parameters(), lr=self.opt_dict['s_given_c_lr'])
                    self.content_decoder_optimizer = optim.SGD(self.vae.content_decoder.parameters(),
                                                                lr=self.opt_dict['content_decoder_lr'])
                    self.style_classifier1_optimizer = optim.Adam(self.vae.style_classifier1.parameters(), lr=self.opt_dict['style_classifier_lr'])
                    self.style_classifier2_optimizer = optim.Adam(self.vae.style_classifier2.parameters(), lr=self.opt_dict['style_classifier_lr'])
            else:
                self.opt_dict["not_improved"] = 0
                self.opt_dict["best_loss"] = total_loss
            if decay_cnt == 3:
                break

            self.logging('-' * 75)
            self.logging('| end of epoch {:2d} | time {:5.4f}s | '
                         'kl_weight {:.4f} | beta_weight {:.4f} | lr {:.7f}'.format(
                             epoch, (time.time() - epoch_start_time),
                             self.kl_weight, self.beta_weight, self.opt_dict["enc_lr"]))
            self.logging('| Validation: total_loss {:3.4f} | recon {:3.4f} | kl {:3.4f} | dient {:3.4f} | content_recon {:3.4f} | style_pred {:3.4f} | s_c_disent {:3.4f}'.format(
                val_losses[0]+val_losses[3], val_losses[1], val_losses[2], val_losses[3], val_losses[4], val_losses[5], val_losses[6]))
            self.logging('-' * 75)
        
        return best_loss

    def save(self, path):
        self.logging("saving to %s" % path)
        model_path = os.path.join(path, "model.pt")
        torch.save(self.vae.state_dict(), model_path)

    def load(self, path):
        model_path = os.path.join(path, "model.pt")
        self.vae.load_state_dict(torch.load(model_path))

class EvaluateVAE:
    def __init__(self, test, test_labels1, test_labels2, load_path, vocab, vae_params):
        super(EvaluateVAE, self).__init__()

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
    
    def eval_style_transfer(self):
        self.vae.eval()
        sent_idx = 0
        total_sent = 0
        df_111 = []
        df_222 = []
        df_221 = []
        df_112 = []
        df_121 = []
        df_212 = []
        
        for i in range(0,20):
            for next_sent_idx in range(sent_idx+1, self.test_labels1[i].size()[0]):
                if total_sent >= 20:
                    break
                if (self.test_labels1[i][sent_idx:sent_idx+1][0] != self.test_labels1[i][next_sent_idx:next_sent_idx+1][0]) and (self.test_labels2[i][sent_idx:sent_idx+1][0] != self.test_labels2[i][next_sent_idx:next_sent_idx+1][0]):
                    print("Sizes:", self.test_data[i].size(), self.test_labels1[i].size(), self.test_labels2[i].size())
                    print("Sizes_Sliced:", self.test_data[i][:, sent_idx:sent_idx+1].size(), self.test_data[i][:, next_sent_idx:next_sent_idx+1].size())

                    original_sentence_11 = ""
                    original_sentence_22 = ""

                    for j in range(self.test_data[i].size()[0]):
                        original_sentence_11 += self.vocab.id2word(self.test_data[i][j, sent_idx:sent_idx+1]) + " "
                        original_sentence_22 += self.vocab.id2word(self.test_data[i][j, next_sent_idx:next_sent_idx+1]) + " "

                    print("11 original_sentence:", original_sentence_11, "sentiment:", self.test_labels1[i][sent_idx:sent_idx+1], "tense:", self.test_labels2[i][sent_idx:sent_idx+1])
                    print("22 original_sentence:", original_sentence_22, "sentiment:", self.test_labels1[i][next_sent_idx:next_sent_idx+1], "tense:", self.test_labels2[i][next_sent_idx:next_sent_idx+1])
                    c1, s1_1, s1_2, _ = self.vae.encode(self.test_data[i][:, sent_idx:sent_idx+1])
                    c2, s2_1, s2_2, _ = self.vae.encode(self.test_data[i][:, next_sent_idx:next_sent_idx+1])

                    transfer_sentence_112 = self.vae.decoder.beam_search_decode(c1, s1_1, s2_2)
                    transfer_sentence_221 = self.vae.decoder.beam_search_decode(c2, s2_1, s1_2)
                    transfer_sentence_111 = self.vae.decoder.beam_search_decode(c1, s1_1, s1_2)
                    transfer_sentence_222 = self.vae.decoder.beam_search_decode(c2, s2_1, s2_2)
                    transfer_sentence_121 = self.vae.decoder.beam_search_decode(c1, s2_1, s1_2)
                    transfer_sentence_212 = self.vae.decoder.beam_search_decode(c2, s1_1, s2_2)

                    df_111 += [" ".join(transfer_sentence_111[0][:-1])]
                    df_222 += [" ".join(transfer_sentence_222[0][:-1])]
                    df_112 += [" ".join(transfer_sentence_112[0][:-1])]
                    df_221 += [" ".join(transfer_sentence_221[0][:-1])]
                    df_121 += [" ".join(transfer_sentence_121[0][:-1])]
                    df_212 += [" ".join(transfer_sentence_212[0][:-1])]
                    
                    print("111 sentence:", " ".join(transfer_sentence_111[0][:-1]))
                    print("222 sentence:", " ".join(transfer_sentence_222[0][:-1]))
                    print("112 sentence:", " ".join(transfer_sentence_112[0][:-1]))
                    print("221 sentence:", " ".join(transfer_sentence_221[0][:-1]))
                    print("121 sentence:", " ".join(transfer_sentence_121[0][:-1]))
                    print("212 sentence:", " ".join(transfer_sentence_212[0][:-1]))
                    total_sent += 1
                    break
        
        import pandas as pd
        pd.options.display.max_colwidth = 100
        df = pd.DataFrame()
        df["OriginalA"] = df_111
        df["OriginalB"] = df_222
        df["A - Sent Swapped"] = df_121
        df["A - Tense Swapped"] = df_112
        df["B - Sent Swapped"] = df_212
        df["B - Tense Swapped"] = df_221
        # print(df)