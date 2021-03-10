"""
Created by Chin-Yun Yu
Updated by Jing-Hua Lin
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio import load
from torchaudio import transforms
import os
import numpy as np
import random
import pandas as pd
import pickle
import mmap
from scipy.io import wavfile
from intervaltree import IntervalTree
from csv import DictReader
from utils import read_MAPS_txt

sz_float = 4  # size of a float
epsilon = 10e-8  # fudge factor for normalization


class MAPS_Dataset(Dataset):
    def __init__(self,
                 folder,
                 hop_size=441,
                 window_size=16384,
                 epoch_size=100000,
                 type='train', 
                 trainsize='100',
                 *,
                 normalize=True):
        sampling_rate = 44100
        self.sr = sampling_rate
        self.hop_size = hop_size
        self.win_size = window_size
        self.normalize = normalize
        self.size = epoch_size
        time_step = hop_size / sampling_rate
        self.time_step = time_step

        self.trainsize = trainsize / 100

        file_ids = list({os.path.splitext(f)[0] for f in os.listdir(folder)})
        # all 210 pieces for training 

        # full training set has 180 pieces
        size_train = round( self.trainsize * 180)
        # full validation set has 30 pieces 
        size_valid = round( self.trainsize * 30)

        file_ids_valid = random.sample( file_ids, size_valid )
        file_ids_train = file_ids
        for i in range(0, size_valid):
            file_ids_train.remove(file_ids_valid[i])

        file_ids_train = random.sample( file_ids_train, size_train )
        file_ids_train.sort()
        file_ids_valid.sort()
        # TDL: use Torch.utils.data.dataset.random_split

        wav = []
        gt = []
            
        if type == 'train':
            print( "", len(file_ids_train), "songs for training")
            for filename in file_ids_train:
                print("-- reading train", filename)
                y, _ = load(os.path.join(folder, filename + '.wav'), normalization=True, channels_first=False)
                y = y.mean(1)
                y = F.pad(y, (window_size // 2, window_size // 2))
                wav += [y]
            self.wav = wav

            for filename, waves in zip(file_ids_train, wav):
                tree = read_MAPS_txt(os.path.join(folder, filename + '.txt'))
                gt += [tree]
            self.gt = gt   

        elif type == 'valid':
            print( "", len(file_ids_valid), "songs for validation")
            for filename in file_ids_valid:
                print("-- reading valid", filename)
                y, _ = load(os.path.join(folder, filename + '.wav'), normalization=True, channels_first=False)
                y = y.mean(1)
                y = F.pad(y, (window_size // 2, window_size // 2))
                wav += [y]
            self.wav = wav

            # time_step = hop_size / sampling_rate
            for filename, waves in zip(file_ids_valid, wav):
                tree = read_MAPS_txt(os.path.join(folder, filename + '.txt'))
                gt += [tree]
            self.gt = gt

        else:
            print('maps dataset type error: please specify type=train/valid')

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        song_id = np.random.randint(0, len(self.wav))
        # segment_idx = np.random.randint(0, len(self.wav[song_id]))
        # return self.wav[song_id][segment_idx], self.gt[song_id][segment_idx]
        segment_idx = np.random.randint(0, len(self.wav[song_id]) - self.win_size)
        y = self.wav[song_id][segment_idx:segment_idx+self.win_size]
        pianoroll = torch.zeros(88)
        tree = self.gt[song_id]

        for note in tree[segment_idx/self.sr:(segment_idx+self.win_size)/self.sr]:
            pianoroll[note.data] = 1
            
        if self.normalize:
            y = y / (y.norm() + epsilon)
            # torch.norm
            # -- default dim=None, keepdim=False
            # dim=int 會回傳 vector norm
            # dim=None input tensor only 2D 時回傳 matrix norm, only 1D 時回傳 vector norm
            # dim=None 時設 keepdim 會失效
            # 之前是：
            # y = y / (y.norm(dim=1, keepdim=True) + epsilon) 

        return y, pianoroll



class MusicNet(Dataset):
    """
    # adapted from MusicNet <http://homes.cs.washington.edu/~thickstn/musicnet.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        mmap (bool, optional): If true, mmap the dataset for faster access times.
        normalize (bool, optional): If true, rescale input vectors to unit norm.
        window (int, optional): Size in samples of a data point.
        pitch_shift (int,optional): Integral pitch-shifting transformations.
        jitter (int, optional): Continuous pitch-jitter transformations.
        epoch_size (int, optional): Designated Number of samples for an "epoch"

    """
    # -- string variables for building paths
    url = 'https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz'
    raw_folder = 'raw'
    train_data, train_labels, train_tree = 'train_data', 'train_labels', 'train_tree.pckl'
    test_data, test_labels, test_tree = 'test_data', 'test_labels', 'test_tree.pckl'
    extracted_folders = [train_data, train_labels, test_data, test_labels]
    metafile = 'musicnet_metadata.csv'
           
    # testset = [2303, 1819, 2382]
    validset = [2131, 2384, 1792, 2514, 2567, 1876]

    def __init__(self, root, type='train', trainsize='314', preprocess=False, mmap=False, normalize=True, window=16384, pitch_shift=0,
                 jitter=0., epoch_size=100000, category='all'):
        self.mmap = mmap
        self.normalize = normalize
        self.window = window
        self.pitch_shift = pitch_shift
        self.jitter = jitter
        self.size = epoch_size
        self.m = 128

        # newly added
        self.trainsize = trainsize

        self.root = os.path.expanduser(root)

        metadata = pd.read_csv(os.path.join(root, self.metafile))

        # 320 pieces
        if category == 'all':
            ids = metadata['id'].values.tolist()

        # subset from assigned music category like solo piano, piano quintet
        else:
            idx = metadata.index[metadata['ensemble'] == category]
            ids = metadata.loc[idx]['id'].values.tolist()

        if preprocess:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if type == 'test':
            self.data_path = os.path.join(self.root, self.test_data)
            labels_path = os.path.join(self.root, self.test_labels, self.test_tree)
        else:
            self.data_path = os.path.join(self.root, self.train_data)
            labels_path = os.path.join(self.root, self.train_labels, self.train_tree)

        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)

        self.rec_ids = [k for k in list(self.labels.keys()) if k in ids]   

        # 314 pieces
        if type == 'train':
            self.rec_ids = [i for i in self.rec_ids if i not in self.validset]
            self.rec_ids = random.sample( self.rec_ids, self.trainsize )
            
        # 6 pieces
        elif type == 'valid':
            self.rec_ids = [i for i in self.validset if i in self.rec_ids]

        # entire 320 pieces
        else:
            self.rec_ids = [i for i in self.rec_ids]

        print( "\n", len(self.rec_ids), "songs for dataset -", type )
        self.records = dict()
        self.open_files = []

    def __enter__(self):
        for record in os.listdir(self.data_path):
            if not record.endswith('.npy'): continue
            if self.mmap:
                fd = os.open(os.path.join(self.data_path, record), os.O_RDONLY)
                buff = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
                self.records[int(record[:-4])] = (buff, len(buff) / sz_float)
                self.open_files.append(fd)
            else:
                f = open(os.path.join(self.data_path, record))
                self.records[int(record[:-4])] = (
                    os.path.join(self.data_path, record), os.fstat(f.fileno()).st_size / sz_float)
                f.close()

    def __exit__(self, *args):
        if self.mmap:
            for mm in self.records.values():
                mm[0].close()
            for fd in self.open_files:
                os.close(fd)
            self.records = dict()
            self.open_files = []

    def access(self, rec_id, s, shift=0, jitter=0):
        """
        Args:
            rec_id (int): MusicNet id of the requested recording
            s (int): Position of the requested data point
            shift (int, optional): Integral pitch-shift data transformation
            jitter (float, optional): Continuous pitch-jitter data transformation
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """

        scale = 2. ** ((shift + jitter) / 12.)

        if self.mmap:
            x = np.frombuffer(self.records[rec_id][0][s * sz_float:int(s + scale * self.window) * sz_float],
                              dtype=np.float32).copy()
        else:
            fid, _ = self.records[rec_id]
            with open(fid, 'rb') as f:
                f.seek(s * sz_float, os.SEEK_SET)
                x = np.fromfile(f, dtype=np.float32, count=int(scale * self.window))

        if self.normalize: x /= np.linalg.norm(x) + epsilon

        xp = np.arange(self.window, dtype=np.float32)
        x = np.interp(scale * xp, np.arange(len(x), dtype=np.float32), x).astype(np.float32)

        y = np.zeros(self.m, dtype=np.float32)
        for label in self.labels[rec_id][s + scale * self.window / 2]:
            y[label.data[1] + shift] = 1

        return x, y
        

    def __getitem__(self, index):
        """
        Args:
            index (int): (ignored by this dataset; a random data point is returned)
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """

        shift = 0
        if self.pitch_shift > 0:
            shift = np.random.randint(-self.pitch_shift, self.pitch_shift)

        jitter = 0.
        if self.jitter > 0:
            jitter = np.random.uniform(-self.jitter, self.jitter)

        rec_id = self.rec_ids[np.random.randint(0, len(self.rec_ids))]
        s = np.random.randint(0, self.records[rec_id][1] - (2. ** ((shift + jitter) / 12.)) * self.window)
        return self.access(rec_id, s, shift, jitter)

    def __len__(self):
        return self.size

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_data)) and \
               os.path.exists(os.path.join(self.root, self.test_data)) and \
               os.path.exists(os.path.join(self.root, self.train_labels, self.train_tree)) and \
               os.path.exists(os.path.join(self.root, self.test_labels, self.test_tree))

    def download(self):
        """Download the MusicNet data if it doesn't exist in ``raw_folder`` already."""

        # process and save as torch files
        print('Processing...')

        self.process_data(self.test_data)

        trees = self.process_labels(self.test_labels)
        with open(os.path.join(self.root, self.test_labels, self.test_tree), 'wb') as f:
            pickle.dump(trees, f)

        self.process_data(self.train_data)

        trees = self.process_labels(self.train_labels)
        with open(os.path.join(self.root, self.train_labels, self.train_tree), 'wb') as f:
            pickle.dump(trees, f)

        print('Download Complete')

    # write out wavfiles as arrays for direct mmap access
    def process_data(self, path):
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.wav'): continue
            uid = int(item[:-4])
            _, data = wavfile.read(os.path.join(self.root, path, item))
            np.save(os.path.join(self.root, path, item[:-4]), data)

    # wite out labels in intervaltrees for fast access
    def process_labels(self, path):
        trees = dict()
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.csv'): continue
            uid = int(item[:-4])
            tree = IntervalTree()
            with open(os.path.join(self.root, path, item), 'r') as f:
                reader = DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note'])
                    start_beat = float(label['start_beat'])
                    end_beat = float(label['end_beat'])
                    note_value = label['note_value']
                    tree[start_time:end_time] = (instrument, note, start_beat, end_beat, note_value)
            trees[uid] = tree
        return trees

