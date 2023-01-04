import nltk
import random
import numpy as np
import torch
from simple_transformer.trainer import Trainer
from simple_transformer.model import Model
from simple_transformer.data import Data
from simple_transformer.tokeniser import Tokeniser
import argparse
import wandb

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', action='store_true')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--window-size', type=int, default=64)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    return parser.parse_args()

args = parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print('Reading text...')
text = nltk.corpus.gutenberg.raw('austen-emma.txt')
print(f'Preparing data...')

text = text[:1000] # TEMP: Use a small subset of the data for testing

data = Data(text, args.window_size)
model = Model(Tokeniser().vocab_size(), args.embedding_dim)

wandb.init(
    mode='online' if args.track else 'disabled',
    project='simple-transformer',
    name=args.exp_name,
    tags=args.tags,
    config={
        'seed': args.seed,
        'num_epochs': args.epochs,
        'lr': args.lr,
    }
)

wandb.watch(model)

trainer = Trainer(model, wandb, lr=args.lr, bs=args.batch_size)

print('Training...')
trainer.train(args.epochs, data)

model.save('model.pt')
