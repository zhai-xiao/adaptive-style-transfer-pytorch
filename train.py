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

from utils import prepare_sub_folder, write_loss, get_config, write_2images, normalize_arr_of_imgs, denormalize_arr_of_imgs
import argparse
from model import ArtGAN
import torch
import os
import tensorboardX
import shutil
import prepare_dataset
import img_augm
from tqdm import tqdm
import multiprocessing
import time
from collections import namedtuple

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/vangogh.yaml', help='path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(options.config))[0]
    train_writer = tensorboardX.SummaryWriter(os.path.join(options.output_path + "/logs", model_name))
    output_directory = os.path.join(options.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(options.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

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
    trainer = ArtGAN(opts).cuda()
    initial_step = trainer.resume(checkpoint_directory, opts) if options.resume else 0
    # prepare data
    augmentor = img_augm.Augmentor(crop_size=[opts.image_size, opts.image_size],
                                    vertical_flip_prb=0.,
                                    hsv_augm_prb=1.0,
                                    hue_augm_shift=0.05,
                                    saturation_augm_shift=0.05, 
                                    saturation_augm_scale=0.05,
                                    value_augm_shift=0.05, 
                                    value_augm_scale=0.05, )
    content_dataset_places = prepare_dataset.PlacesDataset(opts.content_data_path)
    art_dataset = prepare_dataset.ArtDataset(opts.art_data_path)
    q_art = multiprocessing.Queue(maxsize=10)
    q_content = multiprocessing.Queue(maxsize=10)
    jobs = []
    for i in range(4):
        p = multiprocessing.Process(target=content_dataset_places.initialize_batch_worker,
                                    args=(q_content, augmentor, opts.batch_size, i))
        p.start()
        jobs.append(p)

        p = multiprocessing.Process(target=art_dataset.initialize_batch_worker,
                                    args=(q_art, augmentor, opts.batch_size, i))
        p.start()
        jobs.append(p)
    print("Processes are started.")
    time.sleep(3)
    # config training
    win_rate = opts.discr_success_rate
    discr_success = opts.discr_success_rate
    alpha = 0.05
    test = torch.cat([torch.tensor(q_content.get()['image'], requires_grad=False) for i in range(opts.display_size)])    
    test = normalize_arr_of_imgs(test.cuda()).permute(0,3,1,2)
    torch.backends.cudnn.benchmark = True
    # Start training
    for step in tqdm(range(initial_step, opts.max_iter+1), initial=initial_step, total=opts.max_iter, ncols=64, mininterval = 2):
        # Get batch from the queue with batches q, if the last is non-empty.
        while q_art.empty() or q_content.empty():
            pass
        batch_art = normalize_arr_of_imgs(torch.tensor(q_art.get()['image'], requires_grad=False).cuda()).permute(0,3,1,2).requires_grad_()
        batch_content = normalize_arr_of_imgs(torch.tensor(q_content.get()['image'], requires_grad=False).cuda()).permute(0,3,1,2).requires_grad_()
        # Training update
        trainer.update_learning_rate()
        discr_success = trainer.update(batch_art, batch_content, opts, discr_success, alpha, discr_success >= win_rate)

        # Dump training stats in log file
        if step % 10 == 0:
            write_loss(step, trainer, train_writer)
        # Save network weights
        if (step+1) % opts.save_freq == 0:
            trainer.save(checkpoint_directory, step)
        if step % 50 == 0:
            print("Iteration: %08d/%08d, dloss = %.8s, gloss = %.8s, discr_success = %.5s" % (step, opts.max_iter, trainer.discr_loss.item(), trainer.gener_loss.item(), discr_success))
        # Write images
        if (step+1) % 100 == 0:
            del batch_art, batch_content
            torch.cuda.empty_cache()
            with torch.no_grad():
                samp = trainer.sample(test)
                image_outputs = [denormalize_arr_of_imgs(samp[0]), denormalize_arr_of_imgs(samp[1])]
            write_2images(image_outputs, opts.display_size, image_directory, 'test_%08d' % (step + 1))
            del samp, image_outputs
            torch.cuda.empty_cache()

    print("Training is finished. Terminate jobs.")
    for p in jobs:
        p.join()
        p.terminate()
    print("Done.")
