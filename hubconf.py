dependencies = ['torch']
import torch
from torch import nn
import netvlad
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg16_netvlad(pretrained=False):
    encoder = models.vgg16(pretrained=pretrained)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)
    net_vlad = netvlad.NetVLAD(num_clusters=64, dim=512, vladv2=False)
    model.add_module('pool', net_vlad)

    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/checkpoint.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(resume_ckpt['state_dict'])
    model = model.to(device)
    model.eval()    

    return model

def vgg16_netvlad_flip_v_and_h(pretrained=False):
    encoder = models.vgg16(pretrained=pretrained)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)
    net_vlad = netvlad.NetVLAD(num_clusters=64, dim=512, vladv2=False)
    model.add_module('pool', net_vlad)

    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/vgg16_netvlad_rot_query.pth.tar', map_location=torch.device('cpu'))
    print(resume_ckpt)
    model.load_state_dict(resume_ckpt['state_dict'])

    model = model.to(device)
    model.eval()    

    return model