import torch
from jetson_nano.config import cfg as jn_cfg
from torch2trt import TRTModule


# Load trt model
model_trt_path = jn_cfg['MODEL_PATH']
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(model_trt_path))


# Retrieve data


# Inference


# Output
