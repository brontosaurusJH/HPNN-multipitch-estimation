import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torchaudio import load
from torchaudio import save
import argparse
import os
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from dataset import epsilon
from acoustics.generator import pink
import pickle
from time import time
from datetime import timedelta
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='Datasets/musicnet/')
parser.add_argument('--model_path', type=str, default='Models/musicnet_aug_6L.pth')
parser.add_argument('--framesize', type=int, default=16384)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--snr', type=float, help='pink noise in dB.')


def _noise_scaler(signal_power, snr):
    return (signal_power / 10 ** (snr / 10)).sqrt()
 

if __name__ == '__main__':
    args = parser.parse_args()

    sr = 44100
    hopsize = 512
    frame_size = args.framesize
    batch_size = args.batch
    test_len = 90 * sr // hopsize + 1
    test_ids = [2303, 1819, 2382]

    t_start = time()
    net = torch.load(args.model_path)
    net.eval().cuda()

    test_data = []
    test_labels = []
    valid_data = []
    valid_labels = []

    with open(os.path.join(args.root, 'test_labels', 'test_tree.pckl'), 'rb') as f:
        trees = pickle.load(f)

    for id in test_ids:
        print("\n -- loading", id)
        audio_path = os.path.join(args.root, 'test_data', str(id) + '.wav')
        # print( ' ', audio_path)
        y, sr = load(audio_path, normalization=True, channels_first=False)
        y = y.mean(1)
        y = F.pad(y, (frame_size // 2, frame_size // 2))
        y = y[:len(y) // hopsize * hopsize].unfold(0, frame_size, hopsize)

        if args.snr is None:
            pass
        else:
            noise = torch.Tensor(pink(frame_size))
            data_pow = y.pow(2).mean(1)
            scaler = _noise_scaler(data_pow, args.snr)
            noise = scaler[:, None] * noise
            y += noise
            print("  contaminated with pink noise at SNR = " + str(args.snr))

        y = y / (y.norm(dim=1, keepdim=True) + epsilon)

        label = torch.zeros(y.size(0), 88)
        label_tree = trees[id]
        for i in range(y.size(0)):
            for note in label_tree[i * hopsize]:
                label[i, note.data[1] - 21] = 1

        if y.shape[0] > label.shape[0]:
            y = y[:label.shape[0]]
        elif y.shape[0] < label.shape[0]:
            y = F.pad(y, (0, 0, 0, label.shape[0] - y.shape[0]))

        test_data.append(y[:test_len])
        valid_data.append(y[test_len:])
        test_labels.append(label[:test_len])
        valid_labels.append(label[test_len:])

    test_data = torch.cat(test_data, 0)
    test_labels = torch.cat(test_labels, 0)
    valid_data = torch.cat(valid_data, 0)
    valid_labels = torch.cat(valid_labels, 0)

    valid_set = TensorDataset(valid_data, valid_labels)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=1)
    test_set = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=1)

    y_true = []
    y_score = []

    with torch.no_grad():
        print("\n Finding optimal threshold on valid data ...")
        for _, (inputs, targets) in enumerate(valid_loader):
            inputs = inputs.cuda()
            targets = targets
            y_true += [targets.detach().numpy()]

            outputs = torch.sigmoid(net(inputs))
            y_score += [outputs.detach().cpu().numpy()]
            
    y_score = np.vstack(y_score).flatten()
    y_true = np.vstack(y_true).flatten()

    def threshold(x):
        y2 = y_score > x
        return 1 - f1_score(y_true, y2)

    res = minimize_scalar(threshold, bounds=(0, 1), method='bounded')
    thresh = res.x
    print( " -- threshold: %.4f \n" % thresh )

    y_true = []
    y_score = []

    with torch.no_grad():
        print(" ** Start inference ...")
        for _, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda()
            targets = targets
            y_true += [targets.detach().numpy()]

            outputs = torch.sigmoid(net(inputs))
            y_score += [outputs.detach().cpu().numpy()]

    y_score = np.vstack(y_score).flatten()
    y_true = np.vstack(y_true).flatten()
    print( "    average precision: %.4f" % average_precision_score(y_true, y_score),
           "    precision: %.4f" % precision_recall_fscore_support(y_true, y_score > thresh, average='binary')[0],
           "    recall: %.4f" % precision_recall_fscore_support(y_true, y_score > thresh, average='binary')[1],
           "    f-score: %.4f" % precision_recall_fscore_support(y_true, y_score > thresh, average='binary')[2],
           sep='\n')

    t_cost = time() - t_start
    t_cost = timedelta( seconds=t_cost )

    print("\n -- RunTime: %s \n" % t_cost)