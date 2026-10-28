"""Treina e avalia um modelo. Exemplos:

    python -m src.train --dataset cifar10 --model minhacnn --epochs 40 --tag minhacnn_cifar10
    python -m src.train --dataset cifar10 --model resnet18 --epochs 5 --lr 0.01 --bs 64 \
        --tag resnet18_ft_cifar10

Saídas: results/<tag>/{history.json, metrics.json, test_preds.npz} e checkpoints/<tag>.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .data import CLASSES, ROOT, GPULoader, get_splits, make_augment
from .models import build_model, count_params


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    p.add_argument("--model", choices=["minhacnn", "resnet18"], required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1, help="LR máximo do OneCycle")
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--no-reg", action="store_true",
                   help="ablação: sem data augmentation, dropout, weight decay e label smoothing")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--freeze", action="store_true", help="congela o backbone (só treina a fc)")
    p.add_argument("--img-size", type=int, default=224, help="resolução de entrada da ResNet")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def param_groups(model: nn.Module, wd: float):
    """Weight decay só nos pesos de conv/linear (não em BN e bias)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim > 1 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits, ys = [], []
    for xb, yb in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(xb)
        logits.append(out.float()); ys.append(yb)
    logits, ys = torch.cat(logits), torch.cat(ys)
    loss = F.cross_entropy(logits, ys).item()
    acc = (logits.argmax(1) == ys).float().mean().item()
    return loss, acc, logits, ys


@torch.no_grad()
def throughput(model, x, bs=256, reps=3):
    """Imagens por segundo em inferência (lote 256, fp16)."""
    model.eval()
    xb = x[:bs].float().div(255)
    for _ in range(5):
        with torch.autocast("cuda", dtype=torch.float16):
            model(xb)
    torch.cuda.synchronize()
    t = time.perf_counter()
    n = 0
    for _ in range(reps):
        for k in range(0, len(x), bs):
            xb = x[k:k + bs].float().div(255)
            with torch.autocast("cuda", dtype=torch.float16):
                model(xb)
            n += len(xb)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"

    out_dir = ROOT / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    ckpt = ROOT / "checkpoints" / f"{args.tag}.pt"

    splits = get_splits(args.dataset)
    augment = None if args.no_reg else make_augment(args.dataset)
    train_ld = GPULoader(*splits["train"], args.bs, True, augment, dev)
    train_eval_ld = GPULoader(*splits["train"], 512, False, None, dev)
    val_ld = GPULoader(*splits["val"], 512, False, None, dev)
    test_ld = GPULoader(*splits["test"], 512, False, None, dev)

    model = build_model(args.model, args.dataset, regularized=not args.no_reg,
                        pretrained=not args.no_pretrained, img_size=args.img_size,
                        freeze_backbone=args.freeze).to(dev)
    wd = 0.0 if args.no_reg else args.wd
    ls = 0.0 if args.no_reg else args.label_smoothing
    opt = torch.optim.SGD(param_groups(model, wd), lr=args.lr, momentum=0.9, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(train_ld), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=ls)

    n_params, n_train = count_params(model), count_params(model, trainable_only=True)
    print(f"[{args.tag}] params={n_params:,} treináveis={n_train:,} "
          f"treino={len(train_ld.x)} val={len(val_ld.x)} teste={len(test_ld.x)}", flush=True)

    history, best_acc, best_epoch = [], -1.0, 0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze:  # extração de características: BN do backbone congelado
            model.eval()
        t0 = time.perf_counter()
        sum_loss = torch.zeros((), device=dev); sum_ok = torch.zeros((), device=dev); n = 0
        for xb, yb in train_ld:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(xb)
                loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():  # CE "pura" para comparar com a validação
                sum_loss += F.cross_entropy(out.float(), yb, reduction="sum")
                sum_ok += (out.argmax(1) == yb).sum()
            n += len(yb)
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        val_loss, val_acc, _, _ = evaluate(model, val_ld)
        rec = {"epoch": epoch, "train_loss": sum_loss.item() / n, "train_acc": sum_ok.item() / n,
               "val_loss": val_loss, "val_acc": val_acc, "lr": opt.param_groups[0]["lr"],
               "epoch_time_s": epoch_time}
        history.append(rec)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt)
        print(f"[{args.tag}] ep {epoch:02d}/{args.epochs} "
              f"treino loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {epoch_time:.1f}s", flush=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    train_time = time.perf_counter() - t_start

    # ---------------- avaliação final com o melhor checkpoint (val) ----------------
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    test_loss, test_acc, logits, ys = evaluate(model, test_ld)
    # acurácia "limpa" no treino (sem aumento de dados / dropout) para medir o gap
    clean_train_loss, clean_train_acc, _, _ = evaluate(model, train_eval_ld)
    y_true = ys.cpu().numpy(); probs = logits.softmax(1).cpu().numpy()
    y_pred = probs.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    per_class = (cm.diagonal() / cm.sum(1)).tolist()
    ips = throughput(model, test_ld.x)

    metrics = {
        "tag": args.tag, "dataset": args.dataset, "model": args.model, "config": vars(args),
        "params": n_params, "trainable_params": n_train,
        "best_epoch": best_epoch, "best_val_acc": best_acc,
        "test_acc": test_acc, "test_loss": test_loss,
        "test_precision_macro": float(p), "test_recall_macro": float(r), "test_f1_macro": float(f1),
        "clean_train_acc": clean_train_acc, "clean_train_loss": clean_train_loss,
        "per_class_acc": per_class, "classes": CLASSES[args.dataset],
        "confusion_matrix": cm.tolist(), "train_time_s": train_time,
        "mean_epoch_time_s": float(np.mean([h["epoch_time_s"] for h in history])),
        "inference_img_per_s": ips, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    np.savez_compressed(out_dir / "test_preds.npz", y_true=y_true, y_pred=y_pred,
                        probs=probs.astype(np.float16))
    print(f"[{args.tag}] TESTE acc={test_acc:.4f} f1={f1:.4f} (melhor época {best_epoch}) "
          f"tempo={train_time/60:.1f} min | inferência {ips:.0f} img/s", flush=True)


if __name__ == "__main__":
    main()
