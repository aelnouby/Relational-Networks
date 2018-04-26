"""Run script
Author: Alaaeldin Ali"""

from train.trainer import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2.5e-4)
parser.add_argument('--vis_screen', default='Relnet')
parser.add_argument('--save_path', default=None)
parser.add_argument('-warmup', action='store_true')
parser.add_argument('--batch_size', default=64)
args = parser.parse_args()
trainer = Trainer(lr=args.lr,
                  screen=args.screen,
                  batch_size=args.batch_size,
                  save_path=args.save_path,
                  warmup=args.warmup)
trainer.train()
