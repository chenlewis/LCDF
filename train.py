#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import warnings

import numpy as np
import random

from time import sleep
from random import randint
import torch.nn as nn

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer_FMAGgs import Trainer   ###multigpu
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup

import models
import utils
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR, StepLR
import torch
import torchvision
from thop import profile
import torch.distributed as dist

warnings.filterwarnings("ignore")

def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
        print("cur_device",cur_device)
    else:
        cur_device = torch.device('cpu')
    return cur_device
def prepare_training(cfg,logger):
    if cfg.get('resume') is not None:
        print("///resume///")
        sv_file = torch.load(cfg['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
    else:
        print("///no resume///")
        model = models.make(cfg['MODEL'],cfg=cfg).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), cfg['optimizer'])
        epoch_start = 1
    max_epoch = cfg.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=1e-6)
    logger.info('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler



def setup(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    slurm_nodename = os.environ.get("SLURMD_NODENAME", "default_node_name")

    cfg.DIST_INIT_PATH = "tcp://{}:12399".format(slurm_nodename)
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY

    output_folder = os.path.join(
        cfg.DATA.NAME,  f"lr{lr}_wd{wd}")
    # train cfg.RUN_N_TIMES times
    count = 1 ####
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1

    cfg.freeze()
    return cfg


def get_loaders(cfg, logger):
    if cfg.NUM_GPUS > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        # int(os.environ['LOCAL-RANK'])
        torch.cuda.set_device(local_rank)
    logger.info("Loading training data (final training data for vtab)...")
    if cfg.DATA.NAME.startswith("vtab-"):
        train_loader = data_loader.construct_trainval_loader(cfg)
    else:
        train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    # not really needed for vtab
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # main training / eval actions here

    # fix the seed for reproducibility
    # if cfg.SEED is not None:
    #     torch.manual_seed(cfg.SEED)
    #     np.random.seed(cfg.SEED)
    #     random.seed(0)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    logger.info("Constructing models...")

    model, optimizer, epoch_start, lr_scheduler = prepare_training(cfg,logger)
    model.optimizer = optimizer

    cur_device = get_current_device()
    if torch.cuda.is_available():

        if cfg.NUM_GPUS > 1:

            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True)

        else:
            model = model.cuda(device=cur_device)
            print("cur_device",cur_device)

    else:
        model = model.to(cur_device)



    # model, cur_device = build_model(cfg)
    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
    else:
        print("No train loader presented. Exit")

    trainer.eval_classifier(val_loader, "val", save=True)
    trainer.eval_classifier(test_loader, "test", save=True)
    # lr_scheduler.step()


def main(args):
    """main function to call from workflow"""

    # set up cfg and args

    global cfg , log
    cfg = setup(args)
    log, writer = utils.set_save_path(cfg.OUTPUT_DIR, remove=False)
    logger = logging.get_logger("visual_prompt")
    # Perform training.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator()  # Initialize the evaluator
    train(cfg,args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
