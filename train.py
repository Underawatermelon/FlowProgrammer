#!/usr/bin/env python

from config.config import parse_args, create_cfg_from_args
from tools.trainer import Trainer


if __name__ == "__main__":
    args = parse_args()
    cfg = create_cfg_from_args(args)
    trainer = Trainer(cfg)
    trainer.train()
