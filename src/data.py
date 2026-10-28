"""Carregamento dos conjuntos de dados e data loaders que rodam direto na GPU.

- MNIST: arquivos Parquet oficiais do Hugging Face (ylecun/mnist).
- CIFAR-10: versão binária oficial de Toronto (cifar-10-binary.tar.gz).

Os dados ficam inteiros na memória da GPU como uint8 (MNIST ~55 MB, CIFAR ~180 MB),
e o aumento de dados (data augmentation) é feito em lote na GPU. Isso evita
os processos de DataLoader no Windows e deixa o treino bem mais rápido.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

CLASSES = {
    "mnist": [str(i) for i in range(10)],
    "cifar10": ["avião", "automóvel", "pássaro", "gato", "cervo",
                "cachorro", "sapo", "cavalo", "navio", "caminhão"],
}


# --------------------------------------------------------------------------- #
# Leitura dos arquivos brutos
# --------------------------------------------------------------------------- #
def _load_parquet_split(path: Path, col: str):
    """Lê um split em Parquet do Hugging Face (imagens PNG em bytes) -> N,C,H,W uint8."""
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(path)
    imgs = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])), dtype=np.uint8)
                     for d in df[col]])
    imgs = imgs[:, None] if imgs.ndim == 3 else imgs.transpose(0, 3, 1, 2)
    return imgs, df["label"].to_numpy(np.int64)


def _load_cifar_bin(path: Path):
    raw = np.fromfile(path, dtype=np.uint8).reshape(-1, 1 + 3 * 32 * 32)
    return raw[:, 1:].reshape(-1, 3, 32, 32), raw[:, 0].astype(np.int64)


def load_raw(name: str):
    """Retorna (x_train, y_train, x_test, y_test) como arrays uint8/int64 em N,C,H,W."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED / f"{name}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]

    if name == "mnist":
        xtr, ytr = _load_parquet_split(RAW / "mnist" / "train-00000-of-00001.parquet", "image")
        xte, yte = _load_parquet_split(RAW / "mnist" / "test-00000-of-00001.parquet", "image")
    elif name == "cifar10":
        # Os resultados do relatório usam o espelho oficial do Hugging Face (uoft-cs/cifar10).
        # Conferimos que ele contém exatamente as mesmas 50.000 + 10.000 imagens/rótulos da
        # versão binária de Toronto (o teste é idêntico, inclusive na ordem; o treino só está
        # em outra ordem — o que muda qual subconjunto vira validação).
        h = RAW / "cifar10_hf"
        d = RAW / "cifar10" / "cifar-10-batches-bin"
        if (h / "train-00000-of-00001.parquet").exists():
            xtr, ytr = _load_parquet_split(h / "train-00000-of-00001.parquet", "img")
            xte, yte = _load_parquet_split(h / "test-00000-of-00001.parquet", "img")
        else:  # versão binária de cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
            parts = [_load_cifar_bin(d / f"data_batch_{i}.bin") for i in range(1, 6)]
            xtr = np.concatenate([p[0] for p in parts])
            ytr = np.concatenate([p[1] for p in parts])
            xte, yte = _load_cifar_bin(d / "test_batch.bin")
    else:
        raise ValueError(name)

    np.savez(cache, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
    return xtr, ytr, xte, yte


def get_splits(name: str, val_size: int = 5000, seed: int = 0):
    """Divide o treino oficial em treino/validação (fixo pela seed); teste é o oficial."""
    xtr, ytr, xte, yte = load_raw(name)
    perm = np.random.RandomState(seed).permutation(len(xtr))
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    t = lambda a: torch.from_numpy(np.array(a))  # cópia gravável
    return {
        "train": (t(xtr[tr_idx]), t(ytr[tr_idx])),
        "val": (t(xtr[val_idx]), t(ytr[val_idx])),
        "test": (t(xte), t(yte)),
    }


# --------------------------------------------------------------------------- #
# Aumento de dados na GPU (opera em lotes float [0,1], formato B,C,H,W)
# --------------------------------------------------------------------------- #
def random_crop(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Recorte aleatório com padding por reflexão (um deslocamento por imagem)."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = (i[:, None] + torch.arange(H, device=x.device))[:, None, :, None]
    cols = (j[:, None] + torch.arange(W, device=x.device))[:, None, None, :]
    b = torch.arange(B, device=x.device)[:, None, None, None]
    c = torch.arange(C, device=x.device)[None, :, None, None]
    return xp[b, c, rows, cols]


def random_hflip(x: torch.Tensor) -> torch.Tensor:
    mask = torch.rand(x.shape[0], device=x.device) < 0.5
    return torch.where(mask[:, None, None, None], x.flip(3), x)


def random_affine(x: torch.Tensor, max_deg: float = 10, max_shift: float = 0.08,
                  scale=(0.9, 1.1)) -> torch.Tensor:
    """Pequenas rotações/translações/escala (usado no MNIST, onde flip não faz sentido)."""
    B = x.shape[0]
    dev = x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(max_deg)
    s = torch.empty(B, device=dev).uniform_(*scale)
    tx = (torch.rand(B, device=dev) * 2 - 1) * max_shift * 2
    ty = (torch.rand(B, device=dev) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def make_augment(dataset: str):
    if dataset == "cifar10":
        return lambda x: random_hflip(random_crop(x, 4))
    if dataset == "mnist":
        return lambda x: random_affine(x)
    raise ValueError(dataset)


class Preprocess(torch.nn.Module):
    """Normaliza e (opcionalmente) converte para 3 canais e redimensiona.

    Fica fora do modelo para que as visualizações possam usar a imagem em [0,1].
    """

    def __init__(self, mean, std, to_rgb: bool = False, size: int | None = None):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
        self.to_rgb, self.size = to_rgb, size

    def forward(self, x):
        if self.to_rgb and x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if self.size and x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear",
                              align_corners=False, antialias=False)
        return (x - self.mean) / self.std


class GPULoader:
    """Itera em lotes de um tensor uint8 que já está na GPU."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool,
                 augment=None, device="cuda"):
        self.x = x.to(device, non_blocking=True)
        self.y = y.to(device, non_blocking=True)
        self.bs, self.shuffle, self.augment = batch_size, shuffle, augment

    def __len__(self):
        return math.ceil(len(self.x) / self.bs)

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, device=self.x.device) if self.shuffle else \
            torch.arange(n, device=self.x.device)
        for k in range(0, n, self.bs):
            b = idx[k:k + self.bs]
            xb = self.x[b].float().div_(255)
            if self.augment is not None:
                xb = self.augment(xb)
            yield xb, self.y[b]
