import torch
import torch.nn as nn
import torch.nn.functional as F
from .HRNet.hrnet_backbone import HighResolutionNet
from .HRNet.hrnet_config import MODEL_CONFIGS

ALIGN_CORNERS = True

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class HRNet18(nn.Module):
    def __init__(self, pretrained_path=None,norm_layer=None):
        super(HRNet18, self).__init__()
        self.model = HighResolutionNet(MODEL_CONFIGS['hrnet18'],bn_type='torchbn',bn_momentum=0.1)
        # self.model = models.resnet50(pretrained=False,norm_layer=norm_layer)
        weight_init(self)

        if pretrained_path is not None:
            self.initialization(pretrained_path)

    def forward(self, x):
        x = self.model(x)

        return x[0],x[1],x[2],x[3]

    def initialization(self, pretrained_path):
        pretrained_dict = torch.load(pretrained_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        pass

class HRNet32(nn.Module):
    def __init__(self, pretrained_path=None,norm_layer=None):
        super(HRNet32, self).__init__()
        self.model = HighResolutionNet(MODEL_CONFIGS['hrnet32'],bn_type='torchbn',bn_momentum=0.01)
        # self.model = models.resnet50(pretrained=False,norm_layer=norm_layer)
        weight_init(self)

        if pretrained_path is not None:
            self.initialization(pretrained_path)

    def forward(self, x):
        x = self.model(x)
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)

        feats = torch.cat([x[0], x1, x2, x3], 1)
        return feats

    def initialization(self, pretrained_path):
        pretrained_dict = torch.load(pretrained_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

class HRNet48(nn.Module):
    def __init__(self, pretrained_path=None,norm_layer=None):
        super(HRNet48, self).__init__()
        self.model = HighResolutionNet(MODEL_CONFIGS['hrnet48'],bn_type='torchbn',bn_momentum=0.01)
        # self.model = models.resnet50(pretrained=False,norm_layer=norm_layer)
        weight_init(self)

        if pretrained_path is not None:
            self.initialization(pretrained_path)

    def forward(self, x):
        x = self.model(x)
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)

        feats = torch.cat([x[0], x1, x2, x3], 1)
        return feats

    def initialization(self, pretrained_path):
        pretrained_dict = torch.load(pretrained_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)