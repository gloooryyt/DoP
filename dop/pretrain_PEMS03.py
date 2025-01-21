import os
import sys
import random

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.losses import masked_mae

from basicts.utils import load_adj

from .dop_arch import DoPPretrain
from .dop_runner import PretrainRunner
from .dop_data import PretrainingDataset
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "DoPPretrain(PEMS03) configuration"
CFG.RUNNER = PretrainRunner
CFG.DATASET_CLS = PretrainingDataset
CFG.DATASET_NAME = "PEMS03"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 288*7
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 8
BATCH_SIZE_ALL=8

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = random.randint(0,10000000)
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "DoPPretrain"
CFG.MODEL.ARCH = DoPPretrain
adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME + "/adj_mx.pkl", "original")
CFG.MODEL.PARAM = {
    "patch_size":12,
    "in_channel":1,
    "embed_dim":96,
    "num_heads":4,
    "mlp_ratio":4,
    "dropout":0.1,
    "temporal_decoder_1_depth": 2,
    "temporal_decoder_2_depth": 2,
    "adj_mx": adj_mx,
    "used_length": 288*3,
    "mode":"pre-train"
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.001,
    "weight_decay":0,
    "eps":1.0e-8,
    "betas":(0.9, 0.95)
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[50],
    "gamma":0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 150
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE =BATCH_SIZE_ALL
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 0
CFG.TRAIN.DATA.PIN_MEMORY = True

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE_ALL
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 0
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# evluation
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE_ALL
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 0
CFG.TEST.DATA.PIN_MEMORY = True
