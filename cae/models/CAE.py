import torch
import torch.nn as nn
import torchvision
from einops.layers.torch import Rearrange
import numpy as np

class encoder(nn.Module):
    def __init__(self, num_layers=4, latent_channel=32, lc_length=128):
        super().__init__()
        
        resBlockChannels = [4, 16, 32, 64, 128]

        layers = [
            nn.Conv1d(1, resBlockChannels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(resBlockChannels[0])
        ]

        for i in range(num_layers):
            layers.append(BasicBlock(resBlockChannels[i], resBlockChannels[i+1], stride=2))

        self.layers = nn.Sequential(*layers)
        
        self.flat = Rearrange('b c l -> b (c l)', c=resBlockChannels[-1])
        
        self.fc_length_1 = int(np.ceil(lc_length/2**(1+num_layers))*resBlockChannels[-1])
        fc_length_2 = 500
        fc_length_3 = 50

        self.fc1  = nn.Sequential(
            nn.Linear(self.fc_length_1, fc_length_2),
            nn.GELU()
        )
        self.fc2  = nn.Sequential(
            nn.Linear(fc_length_2, fc_length_3),
            nn.GELU()
        )
        self.fc3  = nn.Sequential(
            nn.Linear(fc_length_3, latent_channel),
            nn.GELU()
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class decoder(nn.Module):
    def __init__(self, num_layers=4, latent_channel=32, lc_length=128):
        super().__init__()

        upsampleChannels = [128, 64, 32, 16, 4]
        
        self.expandedLength = int(np.ceil(lc_length/2**(num_layers))*upsampleChannels[0])

        self.fc1 = nn.Sequential(
            nn.Linear(latent_channel, self.expandedLength),
            nn.GELU()
        )
        
        layers = []

        layers.append(Rearrange('b (c l) -> b c l', c=upsampleChannels[0]))
        
        for i in range(num_layers):
            layers.append(
                BasicBlock(
                    upsampleChannels[i],
                    upsampleChannels[i+1],
                    stride=2,
                    network='decoder'
                )
            )
            layers.append(nn.BatchNorm1d(upsampleChannels[i+1]))
            layers.append(nn.GELU())


        layers.append(
            nn.ConvTranspose1d(upsampleChannels[-1], 1, kernel_size=1, stride=1)
        )
    
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layers(x)
        
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channel, channel, stride=1, network='encoder'):
        super().__init__()
        
        if network == 'encoder':
            self.conv1 = nn.Conv1d(in_channel, channel, kernel_size=3, stride=stride, padding=1)
        elif network == 'decoder':
            self.conv1 = nn.ConvTranspose1d(
                in_channel, channel, kernel_size=3, stride=stride, padding=1, output_padding=1
            )
            
        self.norm1 = nn.BatchNorm1d(channel)
        self.gelu  = nn.GELU()

        if network == 'encoder':
            self.conv2 = nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1)
        elif network == 'decoder':
            self.conv2 = nn.ConvTranspose1d(channel, channel, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.BatchNorm1d(channel)

        self.resize = None
        if in_channel != channel:
            if network == 'encoder':
                self.resize = nn.Sequential(
                    nn.Conv1d(in_channel, channel, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(channel)
                )
            elif network == 'decoder':
                self.resize = nn.Sequential(
                    nn.ConvTranspose1d(in_channel, channel, kernel_size=1, stride=stride, output_padding=1),
                    nn.BatchNorm1d(channel)
                )
        

    def forward(self, x):
        iden_x = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        
        if self.resize is not None:
            iden_x = self.resize(iden_x)
            
        x += iden_x

        return self.gelu(x)