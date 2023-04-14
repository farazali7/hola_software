import torch
import torch.nn as nn
import torch.nn.functional as F
from classification.src.config import cfg as cls_cfg
from jetson_nano.config import cfg as jn_cfg
from jetson_nano.utils import get_cyton_data, configure_board, preprocess, \
    get_features, get_features_2, perform_mv
from copy import deepcopy
import os
import time
import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from classification.src.utils.preprocessing import downsample
from sklearn.svm import *
import pickle

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


class CNN_ITER4_2(nn.Module):
    def __init__(self, model_cfg):
        super(CNN_ITER4_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3)
        self.bnormconv = nn.BatchNorm2d(num_features=16)
        self.hidden1 = nn.Linear(144, 512)
        self.bnorm1 = nn.BatchNorm1d(num_features=512)
        self.hidden2 = nn.Linear(512, 256)
        self.bnorm2 = nn.BatchNorm1d(num_features=256)
        self.output = nn.Linear(256, 3)
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

def realtime_inference():
    # Load pretrained model state dict
    # model_path = jn_cfg['PT_MODEL_PATH']
    # model_pth = torch.load(model_path)

    # Params and metrics for trainer (just used for backward compatability here)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f'DEVICE: {device}')

    # model_args = {'dropout': 0}

    # Get base model architecture
    # model = CNN_ITER4_2(model_args)
    # # new_dict = convert_keys(model_pth['state_dict'])
    # model.load_state_dict(model_pth)
    # model.eval().cuda()


    with open('jetson_nano/release_models/svc_2.pkl', 'rb') as r:
        model = pickle.load(r)
    print('DONE LOADING MODEL')


    # Retrieve data
    serial_port = "/dev/ttyUSB0"
    board, emg_channels = configure_board(serial_port)
    emg_channels = emg_channels[:5]
    sample_amount = 300
    conf_threshold = 0.5
    classes = cls_cfg['CLASSES'] + ['NONE']
    try:
        board.prepare_session()
        board.start_stream()
        print('STARTED')
        time.sleep(5)
        buffer = []
        while True:
            time.sleep(0.3)
            data = get_cyton_data(board, sample_amount)
            
            # emg_data = data[emg_channels]
            for count, channel in enumerate(emg_channels):
                DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(data[channel], 250, 3.0, 100.0, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandstop(data[channel], 250, 48.0, 52.0, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandstop(data[channel], 250, 58.0, 62.0, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)

            emg_data = np.transpose(data[emg_channels])
            
            # preprocessed_data = preprocess(emg_data)
            # del emg_data
            # emg_data = downsample(emg_data, 200, 250)
            emg_data = np.abs(emg_data)

            features = get_features_2(emg_data)
            del emg_data

            # features = torch.Tensor(features).to(device)

            # print('---------- PREDICTING ----------')
            outputs = model.predict_proba(features)
            del features
            pred_ids = torch.tensor(np.argmax(outputs, 1))
            pred_probs = torch.tensor(np.max(outputs, 1))
            pred_ids = torch.where(pred_probs < conf_threshold, 3, pred_ids)

            # print(f"I THINK ITS: {outputs}")

            # pred_probs, pred_ids = torch.max(outputs.detach(), 1)
            # pred_ids = torch.where(pred_probs < conf_threshold, 3, pred_ids)
            # print('Done thresholding.')
            pred_grasp_num = perform_mv(torch.tensor(pred_ids))
            # buffer.append(pred_grasp_num)
            # if len(buffer) == 2:
            #     print('---------- PREDICTING ----------')
            #     if buffer.count(buffer[0])==len(buffer):
            #         print(f'PREDICTED GRASP: {classes[buffer[0]]}')
            #     else:
            #         print("NO GRASP. NEED 3 CONSISTENT SIGNALS.")
            #     buffer = []
            print(f'GRASPNUM: {pred_grasp_num}')
            pred_grasp = classes[pred_grasp_num]
            print(f'PREDICTED GRASP: {pred_grasp}')
            print('---------------------------------')
    except Exception as e:
        print(e)
    except KeyboardInterrupt as e:
        print(e)
    finally:
        board.stop_stream()
        board.release_session()

    # Inference


    # Output
    print('Done inferencing.')


if __name__ == "__main__":
    realtime_inference()
