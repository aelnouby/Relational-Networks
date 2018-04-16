from train.trainer import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2.5e-4, type=float)
parser.add_argument('--screen', default='Relnet')
parser.add_argument('--batch_size', default=640, type=int)
args = parser.parse_args()
trainer = Trainer(lr=args.lr, screen=args.screen, batch_size=args.batch_size)
trainer.train()