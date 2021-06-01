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

    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/checkpoint.pth.tar')
    checkpoint = torch.load(resume_ckpt)
    #model.load_state_dict(checkpoint['state_dict'])
    #model = model.to(device)
    #model.eval()    

    return model