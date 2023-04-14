import torch
import torch.nn as nn
import torch.onnx

from copy import deepcopy

from jetson_nano.config import cfg as jn_cfg


class CNN_ITER4(nn.Module):
    def __init__(self, model_cfg):
        super(CNN_ITER4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3)
        self.bnormconv = nn.BatchNorm2d(num_features=16)
        self.hidden1 = nn.Linear(144, 256)
        self.bnorm1 = nn.BatchNorm1d(num_features=256)
        self.hidden2 = nn.Linear(256, 128)
        self.bnorm2 = nn.BatchNorm1d(num_features=128)
        self.output = nn.Linear(128, 3)
        self.output_activation = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(model_cfg['dropout'])

    def forward(self, x):
        # x = torch.reshape(x, (x.shape[0], 5, 5, x.shape[-1]))
        x = torch.permute(x, (0, 3, 2, 1))  # Set channels to dim 1
        x = self.conv1(x)
        x = self.bnormconv(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)

        x = self.hidden1(x)
        x = self.bnorm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = self.bnorm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.output(x)
        x = self.output_activation(x)

        return x
    

# Load pretrained model state dict
model_path = jn_cfg['PT_MODEL_PATH']
model_pth = torch.load(model_path)
print('LOADED PTH')
# Params and metrics for trainer (just used for backward compatability here)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_args = {'dropout': 0}

# Get base model architecture
model = CNN_ITER4(model_args)

new_state_dict = {}
for key in model_pth['state_dict']:
    new_key = key.split('model.')[1]
    new_state_dict[new_key] = deepcopy(model_pth['state_dict'][key])

model.load_state_dict(new_state_dict)
model.eval().cuda()

print('LOADED MODEL')

# create example data
x = torch.ones((3, 5, 5, 5)).cuda()

# Export to ONNX
torch.onnx.export(model, 
                  x, 
                  jn_cfg['ONNX_MODEL_PATH'], 
                  verbose=False)
print("DONE EXPORTING")
