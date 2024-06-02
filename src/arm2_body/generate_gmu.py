import numpy as np
#from collections import namedtuple
import cv2
import os

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import mujoco
import mediapy as media
from PIL import Image
import torchvision.transforms as transforms
import cv2
from decoder_2joint import RGBDecoder


import mujoco
import matplotlib.pyplot as plt
import time
import itertools
from typing import Callable, NamedTuple, Optional, Union, List
import mujoco.viewer
import csv
import pandas as pd

import ffmpeg
import matplotlib.animation as animation
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip



def minmax_normalization(x_input , max_val, min_val):
    # Minmax normalization to get an input x_val in [-1,1] range
    return 2.0*(x_input-min_val)/(max_val-min_val)-1.0




mu = np.zeros((10000,2))
mu1 = np.linspace(-0.698, 0.698, 100)
mu2 = np.linspace(-0.698, 0.698, 100)
mu_uni = np.empty((1, 2))
for i in range(100):
    for j in range(100):
        mu[100*i+j][0] = mu1[i]
        mu[100*i+j][1] = mu2[j]


# CSVファイルのパス
csv_file_path = 'train_data/train_internal/train.csv'

# CSVファイルをDataFrameに読み込む
df = pd.read_csv(csv_file_path, header=None)

# 各列の最大値と最小値を取得
max_values = df.max()
min_values = df.min()

data_max = (np.array(max_values))
data_min = (np.array(min_values))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = RGBDecoder()
net = net.to(device)
model_path = 'net_1000.prm'

params = torch.load(model_path, map_location="cpu")
net.load_state_dict(params)
net.eval()

for i in range(10000):
    mu_uni[0] = mu[i]
    input = torch.FloatTensor(mu_uni).to(device)
    input = Variable(input, requires_grad=True)
    out = net.forward(input)
    g_mu = out.cpu().data.numpy()
    g_mu = np.squeeze(g_mu)
    g_mu = np.transpose(g_mu, (1, 2, 0))
    plt.imsave("gmu_image/%d.png" % (i+1), g_mu)

