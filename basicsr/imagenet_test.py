# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# Copyright (c) 2022 megvii-model
# ------------------------------------------------------------------------


"""
TODO :
<MUST>
Create a testing script on Imagenet:
  [ ] Need to create an image loader for loading DTU images 
  [ ] Apply the pretrained model to do inferencing on the DTU images
  [ ] Compare PSNR between denoised image and noisy image, and save the PSNR data to a file 
"""

import argparse
import datetime
import logging
import math
import random
import time
import torch
import pdb

from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.imagenet_dataset import Imagenet
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--input_path', type=str, required=False, help='The path to the input testing image.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output testing image. ')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    print('Disable distributed.', flush=True)
    opt['rank'], opt['world_size'] = get_dist_info()
    seed = random.randint(1, 10000)
    seed = opt.get('manual_seed')
    set_random_seed(seed + opt['rank'])
    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': getattr(args, 'input_path', '/root/autodl-tmp/nafnet_testin'),
            'output_img': getattr(args, 'output_path', '/root/autodl-tmp/nafnet_testout')
        }
    return opt

if __name__ == '__main__':
    opt = parse_options()
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    
