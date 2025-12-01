#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import warnings
import numpy as np
import random

from time import sleep
from random import randint
import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer_FMAGs import Trainer
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup

import models
import utils
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR, StepLR
import torch
import torchvision
from thop import profile
import torch.distributed as dist
from collections import OrderedDict
import random
from difflib import get_close_matches
import cv2

warnings.filterwarnings("ignore")


def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def prepare_training(cfg, logger):
    if cfg.get('resume') is not None:
        print("///resume///")
        sv_file = torch.load(cfg['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
    else:
        print("///no resume///")
        model = models.make(cfg['MODEL'], cfg=cfg).cuda()
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

    cfg.MODEL.ONLYTEST = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    # cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, f"lr{lr}_wd{wd}")

    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        sleep(randint(3, 5))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        print(output_path)
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg


def get_loaders(cfg, logger):
    # if cfg.NUM_GPUS > 1:
    #     dist.init_process_group(backend='nccl', init_method='env://')
    #     local_rank = int(os.environ['LOCAL_RANK'])
    #     # int(os.environ['LOCAL-RANK'])
    #     torch.cuda.set_device(local_rank)
    logger.info("Loading training data (final training data for vtab)...")
    if cfg.MODEL.ONLYTEST:
        logger.info("...no train data is constructed")
        train_loader = None
    else:
        if cfg.DATA.NAME.startswith("vtab-"):
            train_loader = data_loader.construct_trainval_loader(cfg)
        else:
            train_loader = data_loader.construct_train_loader(cfg)
    if cfg.MODEL.ONLYTEST:
        logger.info("...no validate data is constructed")
        val_loader = None
    else:
        logger.info("Loading validation data...")
        # not really needed for vtab
        val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    test_loader = data_loader.construct_test_loader(cfg)
    return train_loader, val_loader, test_loader


def test(cfg, args):
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

    print("test_main,cfg.MODE.ONLYTEST", cfg.MODEL.ONLYTEST, "trainloader is None", train_loader == None,
          "valloader is none", val_loader == None, "test loader is not none", test_loader != None
          , cfg.MODEL.TESTPTH)
    #
    # model, optimizer, epoch_start, lr_scheduler = prepare_training(cfg, logger)
    # model.optimizer = optimizer
    # lr_scheduler = CosineAnnealingLR(model.optimizer, cfg['epoch_max'], eta_min=1e-6)
    # print(model)
    model = models.make(cfg.MODEL,cfg)
    for k, v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    # print(model)
    cur_device = get_current_device()
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
        # model = model
        # Use multi-process data parallel model in the multi-gpu setting
        if cfg.NUM_GPUS > 1:
            # Make model replica operate on the current device
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device)
            print('111111111111111111')
        print('2222222222222222222222222')
    else:
        model = model.to(cur_device)
        print('33333333333333333333')
    # model, cur_device = build_model(cfg)
    weights_path = cfg.MODEL.TESTPTH

    content = torch.load(weights_path, map_location=torch.device('cpu'))
    # weight = torch.load(weights_path, map_location=torch.device('cpu'))
    # print(weights_path)
    # print(content.keys())
    # model_weights = content['model']
    # for key, value in content.items():
    #     # print(f"Layer: {key}, Shape: {value.shape}")
    #     logger.info(f"Layer: {key}, Shape: {value.shape}")
    # # state_dict = model.state_dict()
    # # logger.info(f"Loading ...,{state_dict}")

    # model.encoder.load_state_dict(content, strict=False)


    state_dict = torch.load(weights_path, map_location='cpu')  # 或者使用 'cuda:0'
    module_lst = [i for i in model.state_dict()]
    weights = OrderedDict()  # 创建一个有序字典,存放权重

    # 用于追踪已匹配的参数
    matched_keys = set()

    # 遍历 state_dict 中的参数
    for k, v in state_dict.items():
        # 寻找与 k 最接近的模型层名称
        close_matches = get_close_matches(k, module_lst, n=1, cutoff=0.4)

        if close_matches:
            best_match = close_matches[0]  # 选择最接近的匹配
            model_param = model.state_dict()[best_match]  # 获取当前模型参数

            # 检查参数形状是否匹配，并且当前键未被匹配
            if model_param.shape == v.shape and best_match not in matched_keys:
                # print(f"匹配层：{k} (模型层：{best_match})，形状：{v.shape}")
                weights[best_match] = v  # 使用 state_dict 中的参数
                matched_keys.add(best_match)  # 标记当前键为已匹配

    # 对于未匹配的层，使用模型参数
    for i in module_lst:
        if i not in matched_keys:
            weights[i] = model.state_dict()[i]  # 如果没有找到匹配的权重，则使用模型中的参数
    model.encoder.load_state_dict(content, strict=False)
    model.eval()
    # logger.info("Loading Encoder from {}".format(cfg.MODEL.TESTPTH))
    logger.info("Setting up Evalutator...")

    logger.info("Constructing models...")
    # model, cur_device = build_model(cfg)

    if cfg.MODEL.ONLYTEST:
        train_loader = None
        val_loader = None

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
    else:
        print("No train loader presented. Exit")

    if cfg.SOLVER.TOTAL_EPOCH == 0 or cfg.MODEL.ONLYTEST:
        print("Test Mode")
        trainer.eval_classifier(test_loader, "test", 0)



def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Only Perform testing.
    test(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
