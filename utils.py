"""
Created in 2017 summer by summer intern Chin Yun Yu
"""
from scipy.io import loadmat
import numpy as np
import pretty_midi
from intervaltree import IntervalTree
from csv import DictReader

from modules import PPower, gamma_layer

def read_midi(file, fs):
    data = pretty_midi.PrettyMIDI(file)
    pianoroll = data.get_piano_roll(fs)
    pianoroll[np.where(pianoroll > 0)] = 1
    return pianoroll[21:109, :]


def read_MAPS_txt(F0):
    tree = IntervalTree()
    with open(F0, 'r') as f:
        reader = DictReader(f, delimiter='\t')
        for note in reader:
            onset_time = float(note['OnsetTime'])
            off_time = float(note['OffsetTime'])
            pitch = int(note['MidiPitch']) - 21
            tree[onset_time:off_time] = pitch

    return tree


def read_bach10_F0s(F0):
    f = np.round(loadmat(F0)['GTF0s'] - 21).astype(int)
    index = np.where(f >= 0)
    pianoroll = np.zeros((88, f.shape[1]))
    for i, frame in zip(index[0], index[1]):
        pianoroll[f[i, frame], frame] = 1
    return pianoroll


def multipitch_evaluation(estimation, truth, raw_value=False):
    if estimation.shape[1] > truth.shape[1]:
        estimation = estimation[:, :truth.shape[1]]
    elif estimation.shape[1] < truth.shape[1]:
        estimation = np.hstack((estimation, np.zeros((88, truth.shape[1] - estimation.shape[1]))))

    TP = np.count_nonzero(truth)
    diff = truth - estimation
    FN = np.where(diff == 1)[0].shape[0]
    FP = np.where(diff < 0)[0].shape[0]
    TP -= FN

    if raw_value:
        return TP, FP, FN
    else:
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        f1 = 2 * p * r / (p + r)
        return p, r, f1


def print_weight(m):
    if type(m) == PPower:
        print("gamma", m.lamda.data[0])
    elif type(m) == gamma_layer:
        print("gamma bias", m.bias.data[0])


if __name__ == '__main__':
    get = read_midi('01-AchGottundHerr.mid', 100)
    print(get.shape)
