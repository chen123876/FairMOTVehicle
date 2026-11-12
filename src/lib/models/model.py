from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.resnet_fpn_dcn import get_pose_net as get_pose_net_fpn_dcn
from .networks.pose_hrnet import get_pose_net as get_pose_net_hrnet
from .networks.pose_dla_conv import get_pose_net as get_dla_conv
from .yolo import get_pose_net as get_pose_net_yolo

_model_factory = {
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'dlaconv': get_dla_conv,
  'resdcn': get_pose_net_dcn,
  'resfpndcn': get_pose_net_fpn_dcn,
  'hrnet': get_pose_net_hrnet,
  'yolo': get_pose_net_yolo
}

def create_model(arch, heads, head_conv):
    if arch.startswith('resdcn'):
        num_layers = 34  # <- 这里改成 34
    else:
        num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
        arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    checkpoint = torch.load(model_path, map_location='cpu')

    # ===== ImageNet pretrained backbone =====
    if 'state_dict' not in checkpoint:
        model.load_state_dict(checkpoint, strict=False)
        print('Loaded ImageNet pretrained weights from {}'.format(model_path))
        return model, optimizer, 0

    # ===== FairMOT checkpoint =====
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    start_epoch = checkpoint.get('epoch', 0)

    if resume and optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Resumed from {}, epoch {}'.format(model_path, start_epoch))
    else:
        print('Loaded model weights from {}'.format(model_path))

    return model, optimizer, start_epoch




def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

