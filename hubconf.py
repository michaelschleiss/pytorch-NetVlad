dependencies = ['torch']
import torch
from torch import nn
import netvlad
from torchvision import models

def vgg16_netvlad(pretrained=False):
    encoder = models.vgg16(pretrained=pretrained)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)
    net_vlad = netvlad.NetVLAD(num_clusters=64, dim=512, vladv2=False)
    model.add_module('pool', net_vlad)
    model.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/yxgeee/OpenIBL/releases/download/v0.1.0-beta/vgg16_netvlad.pth', map_location=torch.device('cpu')))
    return model