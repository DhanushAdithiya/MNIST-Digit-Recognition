import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_TRAIN = 128
BATCH_TEST = 128

OUT_CHANNELS = [128,128]
IN_CHANNELS = 1
LR = 0.001

EPOCH = 100
