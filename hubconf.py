dependencies = ['torch']
import torch
from torch import nn
import netvlad
from torchvision import models

class EmbedNet(nn.Module):
    def __init__(self, dim = 512):
        super(EmbedNet, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        encoder = models.vgg16(pretrained=True)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        #self.add_module('encoder', encoder)
        self.pool = netvlad.NetVLAD(num_clusters=64, dim=dim, vladv2=False)
        #self.pool = nn.AdaptiveMaxPool2d((1,1))
        #self.add_module('pool', net_vlad)
    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        #x = x.view(x.size(0), -1)
        #x =  F.normalize(x, p=2, dim=1)
        return x

class EmbedNetEquiv(nn.Module):
    def __init__(self, dim = 256):
        super(EmbedNetEquiv, self).__init__()
        from backbone import ReResNet
        self.encoder = ReResNet(depth=50)
        #self.add_module('encoder', encoder)
        self.pool = netvlad.NetVLAD(normalize_input=False, num_clusters=64, dim=dim, vladv2=False)
        #self.add_module('pool', net_vlad)
    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return x
    
class EquivNoVlad(nn.Module):
    def __init__(self):
        super(EquivNoVlad, self).__init__()
        from backbone import ReResNet
        self.encoder = ReResNet(depth=50)
        self.pool = nn.AdaptiveMaxPool2d((1,1))
    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        #x =  F.normalize(x, p=2, dim=1)
        return x


def vgg16_netvlad(pretrained=False):
    model = EmbedNet()
    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/vgg16_netvlad.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(resume_ckpt['state_dict'])
    return model

def equiv_netvlad(pretrained=False):
    model = EmbedNetEquiv(dim=256)
    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/equiv_10_epochs.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(resume_ckpt['state_dict'], strict=False)
    return model

def equiv_netvlad_l4(pretrained=False):
    model = EmbedNetEquiv(dim=256)
    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/equiv_17_epochs_l4.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(resume_ckpt['state_dict'], strict=False)
    return model

def equiv_netvlad_l4_no_vlad(pretrained=False):
    model = EquivNoVlad()
    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/equiv_17_epochs_l4.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(resume_ckpt['state_dict'], strict=False)
    return model

def equiv_netvlad_no_gpool_no_vlad(pretrained=False):
    model = EquivNoVlad()
    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/equiv_4_epochs_no_gpool_l4.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(resume_ckpt['state_dict'], strict=False)
    return model

def equiv_netvlad_no_gpool(pretrained=False):
    model = EmbedNetEquiv(dim=2048)
    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/equiv_4_epochs_no_gpool_l4.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(resume_ckpt['state_dict'], strict=False)
    return model

def vgg16_netvlad_imagenet(pretrained=False):
    model = EmbedNet()
    #resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/vgg16_netvlad.pth.tar', map_location=torch.device('cpu'))
    #model.load_state_dict(resume_ckpt['state_dict'])
    
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
    model.load_state_dict(new_state_dict, strict=False)
    #model.load_state_dict(resume_ckpt['state_dict'])

    return model

def vgg16_netvlad_best_180deg(pretrained=False):
    model = EmbedNet()
    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/vgg16_netvlad_model_best_180deg.pth.tar', map_location=torch.device('cpu'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in resume_ckpt['state_dict'].items():
        name = k.replace('module.','') # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    #model.load_state_dict(resume_ckpt['state_dict'])

    return model


def vgg16_netvlad_180deg(pretrained=False):
    model = EmbedNet()
    resume_ckpt = torch.hub.load_state_dict_from_url('https://github.com/michaelschleiss/pytorch-NetVlad/releases/download/v1.0/vgg16_netvlad_checkpoint_180_deg.pth.tar', map_location=torch.device('cpu'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in resume_ckpt['state_dict'].items():
        name = k.replace('module.','') # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    #model.load_state_dict(resume_ckpt['state_dict'])


    return model
