import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt

import os



class RGBDecoder(nn.Module):
    def __init__(self):
        super(RGBDecoder, self).__init__()

        # Two fully connected layers of neurons (feedforward architecture)
        self.ff_layers = nn.Sequential( 
            nn.Linear(2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 16 * 40 * 30),  # 4800 neurons
            nn.LeakyReLU(),
        )

        # Sequential upsampling using the deconvolutional layers & smoothing out checkerboard artifacts with conv layers
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(16, 64, 4, stride=2, padding=1),  # deconv1
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # conv1
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # deconv2
            nn.LeakyReLU(),
            nn.Conv2d(64, 16, 3, stride=1, padding=1),  # conv2
            nn.LeakyReLU(),
            # nn.Dropout(p=0.15),
            # nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),  # deconv3
            # nn.Sigmoid() #Squeezing the output to 0-1 range
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),  # deconv3
            nn.Sigmoid() #Squeezing the output to 0-1 range
        )

        # レイヤーごとに初期化を適用
        for layer in self.ff_layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)  # Kaiming初期化

        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                init.kaiming_normal_(layer.weight)  # Kaiming初期化

        # Sigmoid層にXavier初期化を適用
        for layer in self.output_layer:
            if isinstance(layer, nn.ConvTranspose2d):
                init.xavier_normal_(layer.weight)  # xavier初期化

    def forward(self, x):
        x = self.ff_layers(x)      
        x = x.view(-1, 16, 30, 40)  # Reshaping the output of the fully connected layers to match the conv layers
        x = self.conv_layers(x)
        x = self.output_layer(x)
        return x
    

if __name__=="__main__":


    # Conv_decoderのインスタンスを作成
    net = RGBDecoder()


    net.eval()
    