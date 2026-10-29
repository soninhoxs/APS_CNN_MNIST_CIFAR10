"""Estudo de ablação do aumento de dados (data augmentation) na MinhaCNN.

    python run_ablation.py               # treina o que falta e gera tabelas e figuras
    python run_ablation.py --skip-train  # só refaz a análise (robustez, tabelas, figuras)
    python run_ablation.py --force       # treina tudo de novo

Componente estudado: o aumento de dados. Ligamos e desligamos só ele; dropout, weight
decay, label smoothing, BN, arquitetura, otimizador e número de épocas ficam iguais à
MinhaCNN da APS anterior. Cada configuração principal roda com 3 sementes.

A semente 42 da configuração completa é o próprio treino da APS anterior
(results/minhacnn_mnist e results/minhacnn_cifar10), reaproveitado sem retreinar.

Saídas: results/ablacao/<tag>/, checkpoints/ablacao/<tag>.pt, logs/ablacao/<tag>.log
e, na análise, results/ablacao/*.json|csv|md e figures/ablacao_*.png.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUB = "ablacao"

BASE = {
    "mnist": "--dataset mnist --model minhacnn --epochs 15 --lr 0.1 --wd 5e-4",
    "cifar10": "--dataset cifar10 --model minhacnn --epochs 40 --lr 0.1 --wd 5e-4",
}

EXPERIMENTS = {}
# 1) com x sem aumento de dados, 3 sementes (as execuções "sem" vêm primeiro)
for seed in (42, 43, 44):
    for ds in ("mnist", "cifar10"):
        EXPERIMENTS[f"{ds}_sem_s{seed}"] = f"{BASE[ds]} --no-aug --seed {seed}"
    if seed != 42:  # s42 com aumento = treino da APS anterior
        for ds in ("mnist", "cifar10"):
            EXPERIMENTS[f"{ds}_com_s{seed}"] = f"{BASE[ds]} --seed {seed}"
# 2) CIFAR-10: cada parte do aumento sozinha e intensidade do recorte (semente 42)
EXPERIMENTS["cifar10_flip_s42"] = f"{BASE['cifar10']} --crop-pad 0 --seed 42"   # só espelhamento
EXPERIMENTS["cifar10_crop_s42"] = f"{BASE['cifar10']} --no-flip --seed 42"      # só recorte
for pad in (2, 6, 8):
    EXPERIMENTS[f"cifar10_pad{pad}_s42"] = f"{BASE['cifar10']} --crop-pad {pad} --seed 42"


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
    ap.add_argument("--force", action="store_true", help="retreina mesmo se já houver resultado")
    a = ap.parse_args()
    (ROOT / "logs" / SUB).mkdir(parents=True, exist_ok=True)
    if not a.skip_train:
        for tag, args in EXPERIMENTS.items():
            if a.only and tag not in a.only:
                continue
            if (ROOT / "results" / SUB / tag / "metrics.json").exists() and not a.force:
                print("já existe:", tag, flush=True)
                continue
            run([sys.executable, "-m", "src.train", "--subdir", SUB, "--tag", tag,
                 *args.split()], ROOT / "logs" / SUB / f"{tag}.log")
    if a.train_only:
        return
    run([sys.executable, "-m", "src.ablation"], ROOT / "logs" / SUB / "analise.log")


if __name__ == "__main__":
    main()
