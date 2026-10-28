"""Roda todos os experimentos da APS em sequência e depois gera figuras e relatório.

    python run_all.py              # tudo
    python run_all.py --only minhacnn_mnist resnet18_ft_cifar10
    python run_all.py --skip-train # só regenera figuras/visualizações

Cada experimento roda em um processo separado (libera a memória da GPU) e grava
o log em logs/<tag>.log.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPERIMENTS = {
    # ----- CNN própria (do zero, com regularização) -----
    "minhacnn_mnist": "--dataset mnist --model minhacnn --epochs 15 --lr 0.1 --wd 5e-4",
    "minhacnn_cifar10": "--dataset cifar10 --model minhacnn --epochs 40 --lr 0.1 --wd 5e-4",
    # ablação: mesma CNN sem aumento de dados, dropout, weight decay e label smoothing
    "minhacnn_noreg_cifar10": "--dataset cifar10 --model minhacnn --epochs 40 --lr 0.1 --no-reg",
    # ----- ResNet18 pré-treinada no ImageNet: fine-tuning completo (224x224) -----
    "resnet18_ft_mnist": "--dataset mnist --model resnet18 --epochs 3 --lr 0.01 --wd 1e-4",
    "resnet18_ft_cifar10": "--dataset cifar10 --model resnet18 --epochs 5 --lr 0.01 --wd 1e-4",
    # ----- ResNet18 pré-treinada como extratora de características (backbone congelado) -----
    "resnet18_frozen_mnist": "--dataset mnist --model resnet18 --freeze --epochs 5 --lr 0.1 --wd 1e-4",
    "resnet18_frozen_cifar10": "--dataset cifar10 --model resnet18 --freeze --epochs 5 --lr 0.1 --wd 1e-4",
    # ----- controle: mesma ResNet18 a 224x224, mas sem pesos do ImageNet -----
    "resnet18_scratch_cifar10": "--dataset cifar10 --model resnet18 --no-pretrained --epochs 5 --lr 0.1 --wd 5e-4",
}


def run(cmd, log):
    print(">>", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    tail = Path(log).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
    print("   ", tail[0] if tail else "", flush=True)
    if p.returncode != 0:
        sys.exit(f"falhou: {cmd} (veja {log})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    a = ap.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    if not a.skip_train:
        for tag, args in EXPERIMENTS.items():
            if a.only and tag not in a.only:
                continue
            run([sys.executable, "-m", "src.train", "--tag", tag, *args.split()],
                ROOT / "logs" / f"{tag}.log")
    if a.train_only:
        return
    run([sys.executable, "-m", "src.plots"], ROOT / "logs" / "plots.log")
    run([sys.executable, "-m", "src.activations"], ROOT / "logs" / "activations.log")


if __name__ == "__main__":
    main()
