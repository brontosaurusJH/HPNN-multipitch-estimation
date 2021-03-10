

## HPNN

HPNN stands for **Harmonic preserving neural networks**.

This repository is the code companion with the paper:

Chin-Yun Yu, Jing-Hua Lin and Li Su, "Harmonic preserving neural networks for efficient and robust multipitch estimation," *Proc. Asia Pacific Signal and Infor. Proc. Asso. Annual Summit and Conf. (APSIPA ASC)*, December 2020.




## Dependencies

pytorch 1.0.1 / cuda 10.0 / driver-410
torchaudio 0.2
intervaltree
pretty_midi
python-acoustics



## Training on MusicNet

1. Put MusicNet training data under `<./Datasets/musicnet/train_data>` 
and its labels under `<./Datasets/musicnet/train_labels>` 

2. Start training with the following command:

```
python preprocess.py ./Datasets/musicnet

python musicnet.py --root ./Datasets/musicnet \
                   --out_model your_model.pth
```



## Testing on MusicNet

1. The test set is included in this repository at `<./Datasets/musicnet/test_data>` 
and `<./Datasets/musicnet/test_labels>`

2. The path to HPNN pre-trained model is `<./Models/musicnet_aug_6L.pth>` 

3. Testing with the following command:

```
python test_musicnet.py
```

4. Testing with pink noise degradation:

```
python test_musicnet.py --snr 30
```



### Tasks

- [x] Training / testing on MusicNet
- [ ] Training / testing on MAPS
- [ ] Evaluation in multi-talker scenario


