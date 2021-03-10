# MusicNet provides this data preprocess
# Use this before training on MusicNet
import argparse
from dataset import MusicNet

parser = argparse.ArgumentParser()
parser.add_argument('root', type=str)
args = parser.parse_args()

MusicNet(args.root, preprocess=True)