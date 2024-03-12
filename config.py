import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_TRAIN = 64
BATCH_TEST = 64

OUT_CHANNELS = [128,64]
IN_CHANNELS = 784
LR = 0.001

EPOCH = 100
