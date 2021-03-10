"""
Created by Chin-Yun Yu
Updated by Jing-Hua Lin

"""
import torch
from torch import optim
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from time import time
from datetime import timedelta
from datetime import date

from models import MLC_CFP_pianoroll
from modules import *
from dataset import MusicNet
from utils import print_weight


parser = argparse.ArgumentParser(description='train HPNN on musicnet')

parser.add_argument('--root', type=str, default='../../Dataset/musicnet')
parser.add_argument('--preprocess', action='store_true')
parser.add_argument('--framesize', type=int, default=16384)
parser.add_argument('--out_model', type=str, default='musicnet_aug_6L')
parser.add_argument('--winsize', type=int, default=8192)
parser.add_argument('--hopsize', type=int, default=512)
parser.add_argument('--lr', type=float, default=1E-3)
parser.add_argument('-g', nargs='+', type=float)
# if gamma is specified with -g, gamma will not be trainable
parser.add_argument('--harms_range', type=int, default=25)
parser.add_argument('--steps', type=int, default=16000)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--depth', type=int, default=6)
parser.add_argument('--trainsize', type=int, default=314)
# 314 = 100% of MusicNet training data

def init_weight(m):
    if type(m) == nn.Conv2d:
        N = m.in_channels * np.prod(m.kernel_size)
        m.weight.data.normal_(0., np.sqrt(1 / N))
        if type(m.bias) == torch.Tensor:
            m.bias.data.fill_(0)

def remove_weight_norms(m):
    if hasattr(m, 'weight_g'):
        nn.utils.remove_weight_norm(m)

def add_weight_norms(m):
    if hasattr(m, 'weight'):
        nn.utils.weight_norm(m)


if __name__ == '__main__':

    args = parser.parse_args()

    sr = 44100
    esize = args.steps * args.batch
    batch_size = args.batch
    harms_range = args.harms_range
    frame_size = args.framesize
    winsize = args.winsize
    hopsize = args.hopsize
    num_regions = 1 + (frame_size - winsize) // hopsize
    trainsize = args.trainsize

    mlc_trainable = True
    if args.g:
        g = args.g
        mlc_trainable = False
    else:
        g = [1] * args.depth

    t_start = time()

    print('==> Loading Data...\n')
    # -- without data augmentation
    # train_set = MusicNet(args.root, type='train', trainsize=trainsize, preprocess=args.preprocess, normalize=True, window=frame_size, epoch_size=esize)

    # -- with data augmentation
    train_set = MusicNet(args.root, type='train', trainsize=trainsize, preprocess=args.preprocess, 
                        pitch_transform=5, jitter=.1,
                        normalize=True, window=frame_size, epoch_size=esize)
    valid_set = MusicNet(args.root, type='valid', preprocess=False, normalize=True, window=frame_size,
                        epoch_size=batch_size * 10)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=0)

    print('\n ==> Building model...\n')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = MLC_CFP_pianoroll(winsize, sr, g, hopsize, harms_range, num_regions, mlc_trainable).to(device)
    # net.apply(init_weight)
    # net.apply(add_weight_norms)

    print("    This model has", sum(p.numel() for p in net.parameters() if p.requires_grad), "parameters.\n")
    if device == 'cuda':
        cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12000], gamma=0.2)

    print("==> Start Training.\n")
    print(" step | average_loss | avp_train | average_precision_score(y_true, y_score)")
            
    global_step = 0
    average_loss = []
    avp_train = []
    try:
        with train_set, valid_set:
            net.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets[:, 21:109]

                # scheduler.step()
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # print(global_step, loss.item())
                global_step += 1

                average_loss.append(loss.item())
                y_score = outputs.detach().cpu().numpy().flatten()
                y_true = targets.detach().cpu().numpy().flatten()
                avp_train.append(average_precision_score(y_true, y_score))

                if global_step % 500 == 0:
                    net.eval()
                    with torch.no_grad():
                        y_true = []
                        y_score = []
                        for _, (inputs, targets) in enumerate(valid_loader):
                            inputs = inputs.to(device)
                            targets = targets[:, 21:109]
                            y_true += [targets.detach().numpy()]

                            outputs = net(inputs)
                            y_score += [outputs.detach().cpu().numpy()]

                        y_score = np.vstack(y_score).flatten()
                        y_true = np.vstack(y_true).flatten()
                        print("", global_step, np.mean(average_loss), np.mean(avp_train),
                                average_precision_score(y_true, y_score))
                        average_loss.clear()
                        avp_train.clear()
                    net.train()

        except KeyboardInterrupt:
            print('==> Graceful Exit.\n')
        else:
            print('==> Finish training.\n')
            
            #-- mark ending time
            t_cost = time() - t_start
            t_cost = timedelta( seconds=t_cost )

            print("\n    RunTime: %s \n" % t_cost)

        net.apply(remove_weight_norms)
        net.cpu()
        net = net.module if isinstance(net, torch.nn.DataParallel) else net
        torch.save(net, args.out_model)


