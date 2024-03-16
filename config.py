import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_TRAIN = 1
BATCH_TEST = 1

OUT_CHANNELS = [128,128]
IN_CHANNELS = 1
LR = 0.003

EPOCH = 0
TOTAL_TEST = 15
