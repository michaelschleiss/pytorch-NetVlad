dependencies = ['torch']
import torch
from torch import nn
import netvlad
from torchvision import models

class EmbedNet(nn.Module):
    def __init__(self):
        super(EmbedNet, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        encoder = models.vgg16(pretrained=False)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        #self.add_module('encoder', encoder)
        self.pool = netvlad.NetVLAD(num_clusters=64, dim=512, vladv2=False)
        #self.add_module('pool', net_vlad)
    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return x

def vgg16_netvlad(pretrained=False):
    model = EmbedNet()
    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/vgg16_netvlad.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(resume_ckpt['state_dict'])
    


    return model

def vgg16_netvlad_flip_v_and_h(pretrained=False):
    model = EmbedNet()
    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/vgg16_netvlad_rot_query.pth.tar', map_location=torch.device('cpu'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in resume_ckpt['state_dict'].items():
        name = k.replace('module.','') # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    #model.load_state_dict(resume_ckpt['state_dict'])




    return model