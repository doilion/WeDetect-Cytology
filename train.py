#!/usr/bin/env python
# Copyright (c) WeDetect. All rights reserved.
"""Training script for WeDetect models using MMEngine Runner."""

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it; if not specify, '
        'try to auto resume from the latest checkpoint.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='random seed; overrides cfg.randomness.seed (for multi-seed runs). '
        'default_runtime sets seed=0, so omit this to reproduce the default run.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > config file > default
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp:
        optim_wrapper = dict(cfg.optim_wrapper)
        optim_wrapper.pop('type', None)
        cfg.optim_wrapper = dict(
            type='AmpOptimWrapper',
            constructor='YOLOWv5OptimizerConstructor',
            **optim_wrapper)

    # resume from checkpoint
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # random seed: CLI > config (default_runtime sets randomness=dict(seed=0))
    if args.seed is not None:
        cfg.randomness = dict(cfg.get('randomness', {}))
        cfg.randomness['seed'] = args.seed

    # merge cfg_options
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
