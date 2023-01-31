import nltk
import random
import numpy as np
import torch
from simple_transformer.trainer import Trainer
from simple_transformer.transformer_model import TransformerModel
from simple_transformer.data import Data
from simple_transformer.letter_tokeniser import LetterTokeniser as Tokeniser
import argparse
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', action='store_true')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--truncate-input', type=int, default=None)
    parser.add_argument('--context-size', type=int, default=8)
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-blocks', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval-interval', type=int, default=100)
    return parser.parse_args()

args = parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print('Reading text...')
text = nltk.corpus.gutenberg.raw('austen-emma.txt')
if args.truncate_input is not None:
    text = text[:args.truncate_input]

split = int(len(text) * 0.9)
text_train = text[:split]
text_valid = text[split:]

print(f'Preparing data...')
data_train = Data(text_train, args.context_size)
data_valid = Data(text_valid, args.context_size)

model = TransformerModel(Tokeniser().vocab_size(), args.context_size, args.embedding_dim, args.num_heads, args.num_blocks, args.dropout)
model = model.to(device)

print('Training...')

wandb.init(
    mode='online' if args.track else 'disabled',
    project='simple-transformer',
    name=args.exp_name,
    tags=args.tags,
    config={
        'seed': args.seed,
        'num_epochs': args.epochs,
        'lr': args.lr,
        'context_size': args.context_size,
        'truncate_input': args.truncate_input,
        'batch_size': args.batch_size,
        'embedding_dim': args.embedding_dim,
        'num_heads': args.num_heads,
        'num_blocks': args.num_blocks,
        'dropout': args.dropout,
    }
)
wandb.watch(model)

trainer = Trainer(model, wandb, lr=args.lr, bs=args.batch_size, eval_interval=args.eval_interval)
trainer.train(args.epochs, data_train, data_valid)

model.save('model.pt')
