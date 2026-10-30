"""Declared small-image backbones; no pretrained weights or dataset downloads."""
import copy
import torch
from torch import nn
from torchvision.models import resnet18, mobilenet_v2
from torchvision.models.vgg import make_layers, cfgs

LEVELS = (0, 5, 10, 15, 20, 25, 30, 40)


class SwitchBN(nn.BatchNorm2d):
    """Separate affine parameters and running statistics at each fixed width."""
    def __init__(self, original):
        super().__init__(original.num_features, eps=original.eps,
                         momentum=original.momentum)
        self.load_state_dict(original.state_dict())
        self.banks = nn.ModuleDict({str(p): copy.deepcopy(original) for p in LEVELS[1:]})
        self.level = 0

    def forward(self, x):
        return super().forward(x) if self.level == 0 else self.banks[str(self.level)](x)


class Backbone(nn.Module):
    def __init__(self, stages, channels, head, classes, exits=False):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = head
        # Created after the backbone to preserve identical common initialization.
        self.exits = nn.ModuleList([nn.Linear(c, classes) for c in channels[:3]]) if exits else nn.ModuleList()

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.head(torch.flatten(self.pool(x), 1))

    def forward_all(self, x):
        out = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.exits):
                out.append(self.exits[i](torch.flatten(self.pool(x), 1)))
        return out + [self.head(torch.flatten(self.pool(x), 1))]

    def forward_exit(self, x, exit_index):
        if exit_index == 'final':
            return self(x)
        index = int(exit_index)
        if not 0 <= index < len(self.exits):
            raise ValueError('Invalid early exit')
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == index:
                return self.exits[i](torch.flatten(self.pool(x), 1))


def make_model(dataset, exits=False, switch_bn=False):
    classes = {'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200}[dataset]
    if dataset == 'fashionmnist':
        stages = []
        inp = 1
        for out in (32, 64, 128):
            stages.append(nn.Sequential(nn.Conv2d(inp, out, 3, padding=1, bias=False),
                          nn.BatchNorm2d(out), nn.ReLU(), nn.MaxPool2d(2)))
            inp = out
        channels = [32, 64, 128]
        head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, classes))
    elif dataset == 'cifar10':
        # VGG-16: 13 convolutions + 3 linear layers; small-image head explicitly declared.
        layers = list(make_layers(cfgs['D'], batch_norm=True).children())
        stages, block = [], []
        for layer in layers:
            block.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                stages.append(nn.Sequential(*block))
                block = []
        channels = [64, 128, 256, 512, 512]
        head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(.5),
                             nn.Linear(512, 512), nn.ReLU(), nn.Dropout(.5), nn.Linear(512, classes))
    elif dataset == 'cifar100':
        base = resnet18(weights=None, num_classes=classes)
        base.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        stages = [nn.Sequential(base.conv1, base.bn1, base.relu, base.layer1),
                  base.layer2, base.layer3, base.layer4]
        channels, head = [64, 128, 256, 512], base.fc
    else:
        base = mobilenet_v2(weights=None, num_classes=classes)
        features = list(base.features.children())
        stages = [nn.Sequential(*features[:4]), nn.Sequential(*features[4:7]),
                  nn.Sequential(*features[7:14]), nn.Sequential(*features[14:])]
        channels, head = [24, 32, 96, 1280], base.classifier
    model = Backbone(stages, channels, head, classes, exits)
    if switch_bn:
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.BatchNorm2d):
                parent, _, key = name.rpartition('.')
                setattr(model.get_submodule(parent), key, SwitchBN(module))
    return model


def set_level(model, level):
    for module in model.modules():
        if isinstance(module, SwitchBN):
            module.level = int(level)
