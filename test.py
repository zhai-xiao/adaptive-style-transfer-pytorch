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

from utils import get_config, write_2images, normalize_arr_of_imgs, denormalize_arr_of_imgs
import argparse
from model import ArtGAN
import torch
from collections import namedtuple
from torchvision import transforms
from torch.utils.data import DataLoader
import data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/vangogh.yaml', help='path to the config file.')
    parser.add_argument('--trained_network', type=str, help="path to the trained network file")
    parser.add_argument('--output_path', type=str, help="outputs path")
    parser.add_argument('--input_path', type=str, help="inputs path")
    options = parser.parse_args()

    # Load experiment setting
    config = get_config(options.config)
    OPTIONS = namedtuple('OPTIONS',
                            'batch_size image_size display_size\
                            max_iter save_freq lr\
                            lr_policy step_size gamma\
                            init\
                            dim gf_dim df_dim \
                            content_data_path \
                            art_data_path \
                            discr_loss_weight \
                            transformer_loss_weight \
                            feature_loss_weight \
                            discr_success_rate')
    opts = OPTIONS._make((
                            config['batch_size'], 
                            config['image_size'], 
                            config['display_size'],
                            config['max_iter'], 
                            config['save_freq'], 
                            config['lr'],
                            config['lr_policy'],
                            config['step_size'],
                            config['gamma'],
                            config['init'],
                            config['dim'],
                            config['ngf'], 
                            config['ndf'],                            
                            config['content_data_path'],
                            config['art_data_path'],
                            config['discr_loss_weight'], 
                            config['transformer_loss_weight'],
                            config['feature_loss_weight'],
                            config['discr_success_rate']
                            ))
    myNet = ArtGAN(opts).cuda()
    initial_step = myNet.resume_eval(options.trained_network)
    torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([transforms.Resize(opts.image_size),
                                    transforms.ToTensor()])
    dataset = data.ImageFolder(options.input_path, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)

    for it, images in enumerate(loader):
        test = normalize_arr_of_imgs(images.cuda().detach())
        with torch.no_grad():
            _, samp = myNet.sample(test)
            image_outputs = [denormalize_arr_of_imgs(samp)]
        write_2images(image_outputs, 1, options.output_path, 'test_%08d' % (it+1))
