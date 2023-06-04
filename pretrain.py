from performer_pytorch import PerformerLM
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper

import argparse
import random
import os
from tqdm import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from functools import reduce
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl

from sophia import SophiaG



GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=1, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=8, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/panglao_human.h5ad', help='Path of data for finetune.')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth', help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetune', help='Finetuned model name.')

args = parser.parse_args()

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every

PATIENCE = 10
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

model_name = args.model_name
ckpt_dir = args.ckpt_dir


# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate model

model = PerformerLM(
    num_tokens = args.bin_num + 2,
    dim = 200,
    depth = 3,
    max_seq_len = SEQ_LEN,
    heads = 5,
    causal = False,
    reversible = False,
    use_scalenorm = True,
    local_attn_heads = 0,
    g2v_position_emb = POS_EMBED_USING,
    generalized_attention = True
)

model = AutoregressiveWrapper(model)
model.cuda()

# prepare sc data
    
class SCDatasetPretrain(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        # rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        # full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        
        return full_seq.cuda()

    def __len__(self):
        return self.data.shape[0]

data = sc.read_h5ad(args.data_path)
data = data.X

acc = []
f1 = []
f1w = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pred_list = pd.Series(['un'] * data.shape[0])

index_train = int(data.shape[0]*0.8)
data_train = data[:index_train]
data_val = data[index_train:]
train_dataset = SCDatasetPretrain(data_train, SEQ_LEN)
val_dataset = SCDatasetPretrain(data_val, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# optimizer

optim = SophiaG(model.parameters(), lr=2e-4, 
                    betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
scaler = GradScaler()

# training

for i in tqdm(range(EPOCHS), mininterval=10., desc='training'):
    model.train()

    # for __ in range(GRADIENT_ACCUMULATE_EVERY):
    with autocast():
        # loss = model(next(train_loader), return_loss = True)
        for index, data_batch in enumerate(tqdm(train_loader)):
            loss = model(data_batch, return_loss = True)
            #print(f'training loss: {loss.item()}')
                
        scaler.scale(loss).backward()
        #print(f'training loss: {loss.item()}')

    print(f'training loss: {loss.item()}')

    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optim)
    scaler.update()
    optim.zero_grad()

    if i % GENERATE_EVERY == 0 and i != 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp, GENERATE_LENGTH)
        output_str = decode_tokens(sample)
        print(output_str)

# save model
print('save model')
checkpoint = {'state_dict': model.state_dict(),'optimizer' :optim.state_dict()}
torch.save(checkpoint, os.path.join(ckpt_dir, 'model_gene_attn.pth'))
