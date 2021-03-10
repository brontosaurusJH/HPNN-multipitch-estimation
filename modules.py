import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class PPower(nn.Module):
    def __init__(self, init=None, trainable=True):
        super().__init__()
        if init:
            self.lamda = Parameter(torch.Tensor([init]), requires_grad=trainable)
        else:
            self.lamda = Parameter(torch.Tensor(1).normal_(1., 0.1), requires_grad=trainable)

    def forward(self, input):
        mask = input.gt(0.).float()
        inv_mask = 1 - mask
        input = input * mask + inv_mask
        return input.pow(self.lamda) - inv_mask


class gamma_layer(nn.Module):
    def __init__(self, gamma=1., filter_idx=1, bias=False, trainable=True):
        super().__init__()
        self.g = PPower(init=gamma, trainable=trainable)
        self.idx = filter_idx
        self.bias = Parameter(torch.zeros(1))
        if not bias or not trainable:
            self.bias.requires_grad = False

    def forward(self, x):
        x = torch.rfft(x, 1, normalized=True, onesided=False)[..., 0]
        x[..., :self.idx] = x[..., -self.idx:] = 0
        return self.g(x + self.bias)


class MLC(nn.Module):
    def __init__(self, in_channels, sr, g, hop_size, Hipass_f=27.5, Lowpass_t=0.24, trainable=True):
        super().__init__()
        self.window_size = in_channels
        self.hop_size = hop_size

        self.window = nn.Parameter(torch.hann_window(in_channels), requires_grad=False)
        self.hpi = int(Hipass_f * in_channels / sr) + 1
        self.lpi = int(Lowpass_t * sr / 1000) + 1
        self.g0 = PPower(init=g[0], trainable=trainable)

        self.num_spec = 1
        self.num_ceps = 0
        layers = []
        for d, gamma in enumerate(g[1:]):
            if d % 2:
                layers.append(gamma_layer(gamma, self.hpi, bias=False, trainable=trainable))
                self.num_spec += 1
            else:
                layers.append(gamma_layer(gamma, self.lpi, bias=False, trainable=trainable))
                self.num_ceps += 1

        self.mlc_layers = nn.ModuleList(layers)

    def forward(self, x):
        x = torch.stft(x, self.window_size, self.hop_size, window=self.window, center=False, normalized=True,
                       onesided=False).pow(2).sum(3).transpose(1, 2)
        spec = self.g0(x)
        ceps = torch.zeros_like(spec)
        for d, layer in enumerate(self.mlc_layers):
            if d % 2:
                spec = layer(ceps)
            else:
                ceps = layer(spec)
        # output shape = (batch, time steps, freq)
        return ceps, spec


class Sparse_Pitch_Profile(nn.Module):
    def __init__(self, in_channels, sr, harms_range=24, division=1, norm=False):
        """

        Parameters
        ----------
        in_channels: int
            window size
        sr: int
            sample rate
        harms_range: int
            The extended area above (or below) the piano pitch range (in semitones)
            25 : though somewhat larger, to ensure the coverage is large enough (if division=1, 24 is sufficient)
        division: int
            The division number for filterbank frequency resolution. The frequency resolution is 1 / division (semitone)
        norm: bool
            If set to True, normalize each filterbank so the weight of each filterbank sum to 1.
        """
        super().__init__()
        step = 1 / division
        # midi_num shape = (88 + harms_range) * division + 2
        # this implementation make sure if we group midi_num with a size of division
        # each group will center at the piano pitch number and the extra pitch range
        # E.g., division = 2, midi_num = [20.25, 20.75, 21.25, ....]
        #       dividion = 3, midi_num = [20.33, 20.67, 21, 21.33, ...]
        midi_num = np.arange(20.5 - step / 2 - harms_range, 108.5 + step + harms_range, step)
        # self.midi_num = midi_num 

        fd = 440 * np.power(2, (midi_num - 69) / 12) 
        # self.fd = fd

        self.effected_dim = in_channels // 2 + 1  
        # // 2 : the spectrum/ cepstrum are symmetric

        x = np.arange(self.effected_dim)
        freq_f = x * sr / in_channels
        freq_t = sr / x[1:]
        # avoid explosion; x[0] is always 0 for cepstrum

        inter_value = np.array([0, 1, 0])
        idxs = np.digitize(freq_f, fd)
 
        cols, rows, values = [], [], []
        for i in range(harms_range * division, (88 + 2 * harms_range) * division):
            idx = np.where((idxs == i + 1) | (idxs == i + 2))[0]
            c = idx
            r = np.broadcast_to(i - harms_range * division, idx.shape)
            x = np.interp(freq_f[idx], fd[i:i + 3], inter_value).astype(np.float32)
            if norm and len(idx):
                # x /= (fd[i + 2] - fd[i]) / sr * in_channels
                x /= x.sum()  # energy normalization

            if len(idx) == 0 and len(values) and len(values[-1]):
                # low resolution in the lower frequency (for spec)/ highter frequency (for ceps),
                # some filterbanks will not get any bin index, so we copy the indexes from the previous iteration
                c = cols[-1].copy()
                r = rows[-1].copy()
                r[:] = i - harms_range * division
                x = values[-1].copy()

            cols.append(c)
            rows.append(r)
            values.append(x)

        cols, rows, values = np.concatenate(cols), np.concatenate(rows), np.concatenate(values)
        self.filters_f_idx = (rows, cols)
        self.filters_f_values = nn.Parameter(torch.tensor(values), requires_grad=False)

        idxs = np.digitize(freq_t, fd)
        cols, rows, values = [], [], []
        for i in range((88 + harms_range) * division - 1, -1, -1):
            idx = np.where((idxs == i + 1) | (idxs == i + 2))[0]
            c = idx + 1
            r = np.broadcast_to(i, idx.shape)
            x = np.interp(freq_t[idx], fd[i:i + 3], inter_value).astype(np.float32)
            if norm and len(idx):
                # x /= (1 / fd[i] - 1 / fd[i + 2]) * sr
                x /= x.sum()

            if len(idx) == 0 and len(values) and len(values[-1]):
                c = cols[-1].copy()
                r = rows[-1].copy()
                r[:] = i
                x = values[-1].copy()

            cols.append(c)
            rows.append(r)
            values.append(x)

        cols, rows, values = np.concatenate(cols), np.concatenate(rows), np.concatenate(values)
        self.filters_t_idx = (rows, cols)
        self.filters_t_values = nn.Parameter(torch.tensor(values), requires_grad=False)
        self.filter_size = torch.Size(((88 + harms_range) * division, self.effected_dim))

    def forward(self, ceps, spec):
        ceps, spec = ceps[..., :self.effected_dim], spec[..., :self.effected_dim]
        batch_dim, steps, _ = ceps.size()
        filter_f = torch.sparse_coo_tensor(self.filters_f_idx, self.filters_f_values, self.filter_size)
        filter_t = torch.sparse_coo_tensor(self.filters_t_idx, self.filters_t_values, self.filter_size)
        ppt = filter_t @ ceps.transpose(0, 2).contiguous().view(self.effected_dim, -1)
        ppf = filter_f @ spec.transpose(0, 2).contiguous().view(self.effected_dim, -1)
        return ppt.view(-1, steps, batch_dim).transpose(0, 2), ppf.view(-1, steps, batch_dim).transpose(0, 2)


class CFP(nn.Module):
    def __init__(self, harms_range, num_regions, division, k1=24, k2=48):
        super().__init__()
        self.harmonics_filter = nn.Sequential(nn.Conv2d(2, k1, (1, harms_range * division + 1), bias=False),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(k1, k2, (num_regions, 1), bias=False),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(k2, 1, (1, division), stride=(1, division)))

    def forward(self, ppt, ppf):
        return self.harmonics_filter(torch.stack((ppt, ppf), 1)).squeeze()


class CFP_MaxPool(nn.Module):
    def __init__(self, harms_range, num_regions, division, k1=24, k2=48):
        super().__init__()
        self.harmonics_filter = nn.Sequential(nn.Conv2d(2, k1, (1, harms_range * division + 1), bias=False),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(k1, k2, (num_regions, 1), bias=False),
                                              nn.ReLU(inplace=True),
                                              nn.MaxPool2d((1, division), stride=(1, division)),
                                              nn.Conv2d(k2, 1, 1))

    def forward(self, ppt, ppf):
        return self.harmonics_filter(torch.stack((ppt, ppf), 1)).squeeze()

