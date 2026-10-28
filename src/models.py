"""Modelos: a CNN própria (MinhaCNN) e a ResNet18 pré-treinada no ImageNet."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .data import (CIFAR_MEAN, CIFAR_STD, IMAGENET_MEAN, IMAGENET_STD, MNIST_MEAN,
                   MNIST_STD, Preprocess)


def conv_bn_relu(cin: int, cout: int, use_bn: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class MinhaCNN(nn.Module):
    """CNN no estilo VGG: 3 blocos [conv-BN-ReLU] x2 + max-pool + Dropout2d,
    seguidos de Global Average Pooling e uma cabeça MLP (ligação com a APS de MLP).

    Entrada 32x32 (CIFAR) -> 16 -> 8 -> 4;  entrada 28x28 (MNIST) -> 14 -> 7 -> 3.
    ~1,2 milhão de parâmetros (cerca de 10x menos que a ResNet18).
    """

    def __init__(self, in_ch: int = 3, num_classes: int = 10, widths=(64, 128, 256),
                 drop2d=(0.1, 0.1, 0.2), drop_fc: float = 0.3, use_bn: bool = True):
        super().__init__()
        blocks, cin = [], in_ch
        for w, p in zip(widths, drop2d):
            blocks.append(nn.Sequential(
                conv_bn_relu(cin, w, use_bn),
                conv_bn_relu(w, w, use_bn),
                nn.MaxPool2d(2),
                nn.Dropout2d(p) if p > 0 else nn.Identity(),
            ))
            cin = w
        self.block1, self.block2, self.block3 = blocks
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cin, 128, bias=not use_bn),
            nn.BatchNorm1d(128) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_fc) if drop_fc > 0 else nn.Identity(),
            nn.Linear(128, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def features(self, x):
        return self.block3(self.block2(self.block1(x)))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


class WithPreprocess(nn.Module):
    """Junta normalização/redimensionamento + rede, para receber imagens em [0,1]."""

    def __init__(self, pre: Preprocess, net: nn.Module):
        super().__init__()
        self.pre, self.net = pre, net

    def forward(self, x):
        return self.net(self.pre(x))


def build_model(name: str, dataset: str, *, regularized: bool = True, pretrained: bool = True,
                img_size: int = 224, freeze_backbone: bool = False) -> nn.Module:
    in_ch = 1 if dataset == "mnist" else 3
    if name == "minhacnn":
        mean, std = (MNIST_MEAN, MNIST_STD) if dataset == "mnist" else (CIFAR_MEAN, CIFAR_STD)
        if regularized:
            net = MinhaCNN(in_ch)
        else:  # ablação: mesma arquitetura, sem dropout (BN mantido p/ estabilidade)
            net = MinhaCNN(in_ch, drop2d=(0, 0, 0), drop_fc=0.0)
        return WithPreprocess(Preprocess(mean, std), net)

    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, 10)
        if freeze_backbone:
            for n, p in net.named_parameters():
                p.requires_grad = n.startswith("fc.")
        pre = Preprocess(IMAGENET_MEAN, IMAGENET_STD, to_rgb=True, size=img_size)
        return WithPreprocess(pre, net)

    raise ValueError(name)


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
