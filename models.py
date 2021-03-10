from modules import *

class MLC_CFP_pianoroll(nn.Module):
    def __init__(self, window_size, sr, g, hop_size, harms_range, num_regions, train_mlc=True):
        super().__init__()
        self.mlc = MLC(window_size, sr, g, hop_size, trainable=train_mlc)
        self.filters = Sparse_Pitch_Profile(window_size, sr, harms_range, division=4, norm=True)
        self.cnn = CFP(harms_range, num_regions, division=4, k1=24, k2=24)

    def forward(self, x):
        ceps, spec = self.mlc(x)
        ppt, ppf = self.filters(ceps, spec)
        outputs = self.cnn(ppt, ppf)
        return outputs