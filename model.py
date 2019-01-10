# Copyright 2018-2019 Xiao Zhai
#
# This file is part of Adaptive Style Transfer, my own implementation of the 
# ECCV 2018 paper A Style-Aware Content Loss for Real-time HD Style Transfer.
#
# Adaptive Style Transfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Adaptive Style Transfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from networks import encoder,decoder,discriminator
from utils import weights_init, get_model_list, get_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ArtGAN(nn.Module):
    def __init__(self, options):
        super(ArtGAN, self).__init__()
        # build model
        self.encoder = encoder(options)
        self.decoder = decoder(options)
        self.discriminator = discriminator(options)
        self.discriminator_weight = {"pred_1": 1.,
                                "pred_2": 1.,
                                "pred_4": 1.,
                                "pred_6": 1.,
                                "pred_7": 1.}
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')
        self.abs = nn.L1Loss(reduction='mean')

        # Setup the optimizers        
        dis_params = list(self.discriminator.parameters())
        gen_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=options.lr, betas=(0.5, 0.999), weight_decay=0.0001, amsgrad=True)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=options.lr, betas=(0.5, 0.999), weight_decay=0.0001, amsgrad=True)
        self.dis_scheduler = get_scheduler(self.dis_opt, options)
        self.gen_scheduler = get_scheduler(self.gen_opt, options)

        # Network weight initialization
        self.apply(weights_init(options.init))
        self.discriminator.apply(weights_init('gaussian'))
        self.gener_loss = torch.tensor(0.)
        self.discr_loss = torch.tensor(0.)

    def forward(self):
        return

    def gen_update(self, batch_content, batch_output, batch_output_preds, options):
        gener_loss = sum([self.loss(pred, torch.ones_like(pred)) * self.discriminator_weight[key] for key, pred in zip(batch_output_preds.keys(), batch_output_preds.values())])
        img_loss = self.mse(F.avg_pool2d(batch_output,kernel_size=10,stride=1), F.avg_pool2d(batch_content,kernel_size=10,stride=1))
        feature_loss = self.abs(self.encoder(batch_output), self.encoder(batch_content))
        self.gener_loss = options.discr_loss_weight * gener_loss + options.transformer_loss_weight * img_loss + options.feature_loss_weight * feature_loss
        self.gener_loss.backward(retain_graph=True)
        self.gen_opt.step()
        del gener_loss, img_loss, feature_loss

        # calculate accuracy
        with torch.no_grad():
            batch_output_gener_acc = sum([(pred>torch.zeros_like(pred)).type_as(pred).mean() * self.discriminator_weight[key] for key, pred in zip(batch_output_preds.keys(), batch_output_preds.values())]).item()
            gener_accuracy = batch_output_gener_acc / float(len(self.discriminator_weight.keys()))
        
        return gener_accuracy

    def dis_update(self, batch_art_preds, batch_content_preds, batch_output_preds, options):
        batch_art_discr_loss = sum([self.loss(pred, torch.ones_like(pred)) * self.discriminator_weight[key] for key, pred in zip(batch_art_preds.keys(), batch_art_preds.values())])
        batch_content_discr_loss = sum([self.loss(pred, torch.zeros_like(pred)) * self.discriminator_weight[key] for key, pred in zip(batch_content_preds.keys(), batch_content_preds.values())])
        batch_output_discr_loss = sum([self.loss(pred, torch.zeros_like(pred)) * self.discriminator_weight[key] for key, pred in zip(batch_output_preds.keys(), batch_output_preds.values())])
        self.discr_loss = options.discr_loss_weight * (batch_art_discr_loss + batch_content_discr_loss + batch_output_discr_loss)
        self.discr_loss.backward()
        self.dis_opt.step()
        del batch_art_discr_loss, batch_content_discr_loss, batch_output_discr_loss

        # calculate accuracy
        with torch.no_grad():
            batch_art_discr_acc = sum([(pred>torch.zeros_like(pred)).type_as(pred).mean() * self.discriminator_weight[key] for key, pred in zip(batch_art_preds.keys(), batch_art_preds.values())]).item()
            batch_content_discr_acc = sum([(pred<torch.zeros_like(pred)).type_as(pred).mean() * self.discriminator_weight[key] for key, pred in zip(batch_content_preds.keys(), batch_content_preds.values())]).item()
            batch_output_discr_acc = sum([(pred<torch.zeros_like(pred)).type_as(pred).mean() * self.discriminator_weight[key] for key, pred in zip(batch_output_preds.keys(), batch_output_preds.values())]).item()
            discr_accuracy = (batch_art_discr_acc+batch_content_discr_acc+batch_output_discr_acc) / float(len(self.discriminator_weight.keys())*3)

        return discr_accuracy

    def update(self, batch_art, batch_content, options, discr_success, alpha, update_generator):
        self.dis_opt.zero_grad()
        self.gen_opt.zero_grad()
        batch_output = self.decoder(self.encoder(batch_content))
        batch_output_preds = self.discriminator(batch_output)
        batch_art_preds = self.discriminator(batch_art)
        batch_content_preds = self.discriminator(batch_content)

        if update_generator:
            g_acc = self.gen_update(batch_content, batch_output, batch_output_preds, options)
            discr_success = discr_success * (1. - alpha) + alpha * (1. - g_acc)
        d_acc = self.dis_update(batch_art_preds, batch_content_preds, batch_output_preds, options)
        discr_success = discr_success * (1. - alpha) + alpha * d_acc
        del batch_output, batch_art_preds, batch_content_preds, batch_output_preds
        return discr_success
        
    def sample(self, test):
        self.eval()
        with torch.no_grad():
            y = self.decoder(self.encoder(test))
        self.train()
        return test, y

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, options):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.encoder.load_state_dict(state_dict['a'])
        self.decoder.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.discriminator.load_state_dict(state_dict['a'])
        # Load optimizers
        last_model_name = get_model_list(checkpoint_dir, "opt")
        state_dict = torch.load(last_model_name)
        self.gen_opt.load_state_dict(state_dict['a'])
        self.dis_opt.load_state_dict(state_dict['b'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, options, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, options, iterations)
        print('Resume from iteration %d' % iterations)
        del state_dict, last_model_name
        torch.cuda.empty_cache()
        return iterations

    def resume_eval(self, trained_generator):
        state_dict = torch.load(trained_generator)
        self.encoder.load_state_dict(state_dict['a'])
        self.decoder.load_state_dict(state_dict['b'])
        return

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'opt_%08d.pt' % (iterations + 1))
        torch.save({'a': self.encoder.state_dict(), 'b': self.decoder.state_dict()}, gen_name)
        torch.save({'a': self.discriminator.state_dict()}, dis_name)
        torch.save({'a': self.gen_opt.state_dict(), 'b': self.dis_opt.state_dict()}, opt_name)
