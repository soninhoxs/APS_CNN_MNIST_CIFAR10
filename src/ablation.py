"""Análise do estudo de ablação do aumento de dados na MinhaCNN.

Lê results/ablacao/ e o treino da APS anterior (que é a semente 42 com aumento de dados),
mede a robustez de cada modelo a transformações do conjunto de teste e gera tabelas e
figuras.

Uso:  python -m src.ablation               # tudo
      python -m src.ablation --no-robust   # reaproveita results/ablacao/robustez.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from . import style
from .data import CLASSES, ROOT, get_splits, make_augment
from .models import build_model
from .plots import DS_NAME, br, finish, spread_labels

RES = ROOT / "results"
ABL = RES / "ablacao"
CKPT = ROOT / "checkpoints"
SEEDS = (42, 43, 44)
DSS = ("mnist", "cifar10")
DEV = "cuda"

COM, SEM, NOREG = "Com aumento de dados", "Sem aumento de dados", "Sem nenhuma regularização"
CFG_NAME = {
    "com": COM, "sem": SEM, "noreg": NOREG,
    "flip": "Só espelhamento", "crop": "Só recorte (±4 px)",
    "pad2": "Recorte ±2 px + espelhamento", "pad6": "Recorte ±6 px + espelhamento",
    "pad8": "Recorte ±8 px + espelhamento",
}
# Paleta validada (scripts/validate_palette.js do skill de dataviz, todos os pares):
# azul e laranja são os slots 1 e 2 da paleta do projeto; verde-água substitui o magenta
# da APS anterior, que ficava perto demais do laranja (ΔE 12,9 < 15).
COLORS = {COM: "#2a78d6", SEM: "#eb6834", NOREG: "#1baf7a"}
PARTIAL = "#9ec5f4"  # aumento parcial: azul claro (mesma família do aumento completo)
TRAIN_RANGE = {  # o que o aumento de dados cobre no treino (para sombrear nos gráficos)
    ("cifar10", "shift"): (0, 4),
    ("mnist", "shift"): (0, 0.08 * 28),          # ±8% da largura = ±2,24 px
    ("mnist", "rot"): (-10, 10),
    ("mnist", "scale"): (0.9, 1.1),
}


# --------------------------------------------------------------------------- #
# Leitura dos resultados
# --------------------------------------------------------------------------- #
def paths(ds, cfg, seed=42):
    if cfg == "com" and seed == 42:  # treino da APS anterior
        return RES / f"minhacnn_{ds}", CKPT / f"minhacnn_{ds}.pt"
    if cfg == "noreg":
        return RES / f"minhacnn_noreg_{ds}", CKPT / f"minhacnn_noreg_{ds}.pt"
    tag = f"{ds}_{cfg}_s{seed}"
    return ABL / tag, CKPT / "ablacao" / f"{tag}.pt"


def read_json(p):
    raw = p.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:  # gravado com a codificação padrão do Windows
        return json.loads(raw.decode("cp1252"))


def load_all():
    keys = [(ds, cfg, s) for ds in DSS for cfg in ("com", "sem") for s in SEEDS]
    keys += [("cifar10", "noreg", 42)]
    keys += [("cifar10", c, 42) for c in ("flip", "crop", "pad2", "pad6", "pad8")]
    runs = {}
    for ds, cfg, s in keys:
        d, ck = paths(ds, cfg, s)
        if not (d / "metrics.json").exists():
            continue
        p = np.load(d / "test_preds.npz")
        runs[(ds, cfg, s)] = {"m": read_json(d / "metrics.json"),
                              "h": read_json(d / "history.json"),
                              "ok": p["y_true"] == p["y_pred"], "ckpt": ck, "dir": d}
    return runs


def seeds_of(runs, ds, cfg):
    return [s for s in SEEDS if (ds, cfg, s) in runs]


def metric(runs, ds, cfg, key):
    return np.array([runs[(ds, cfg, s)]["m"][key] for s in seeds_of(runs, ds, cfg)])


def pm(v, d=2, scale=100):
    """'média ± desvio' com vírgula; desvio omitido com uma semente."""
    v = np.asarray(v) * scale
    return br(v.mean(), d) + (f" ± {br(v.std(ddof=1), d)}" if len(v) > 1 else "")


# --------------------------------------------------------------------------- #
# Tabelas e estatística
# --------------------------------------------------------------------------- #
def summary(runs):
    rows = []
    for (ds, cfg, s), r in sorted(runs.items(), key=lambda kv: (kv[0][0] != "mnist", kv[0][1], kv[0][2])):
        m = r["m"]
        rows.append({
            "dataset": DS_NAME[ds], "configuracao": CFG_NAME[cfg], "semente": s,
            "acc_teste": m["test_acc"], "erro_teste": 1 - m["test_acc"],
            "erros": int((~r["ok"]).sum()), "f1_macro": m["test_f1_macro"],
            "precisao_macro": m["test_precision_macro"], "recall_macro": m["test_recall_macro"],
            "perda_teste": m["test_loss"], "acc_treino_limpo": m["clean_train_acc"],
            "gap_treino_teste": m["clean_train_acc"] - m["test_acc"],
            "melhor_epoca": m["best_epoch"], "epocas": m["config"]["epochs"],
            "tempo_treino_min": m["train_time_s"] / 60, "tempo_epoca_s": m["mean_epoch_time_s"],
        })
    with open(ABL / "resumo_ablacao.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    md = ["| Dataset | Configuração | Sementes | Acurácia teste (%) | Erros (de 10.000) | "
          "F1 macro | Acc. treino (%) | Gap (p.p.) | Perda teste | Melhor época |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in DSS:
        for cfg in ("com", "sem", "noreg", "flip", "crop", "pad2", "pad6", "pad8"):
            ss = seeds_of(runs, ds, cfg) if cfg in ("com", "sem") else \
                ([42] if (ds, cfg, 42) in runs else [])
            if not ss:
                continue
            R = [runs[(ds, cfg, s)] for s in ss]
            g = lambda k: np.array([r["m"][k] for r in R])
            err = np.array([(~r["ok"]).sum() for r in R], float)
            md.append(
                f"| {DS_NAME[ds]} | {CFG_NAME[cfg]} | {len(ss)} | {pm(g('test_acc'))} | "
                f"{pm(err, 0, 1)} | {pm(g('test_f1_macro'), 4, 1)} | {pm(g('clean_train_acc'))} | "
                f"{pm(g('clean_train_acc') - g('test_acc'))} | {pm(g('test_loss'), 3, 1)} | "
                f"{pm(g('best_epoch'), 0, 1)} |")
    (ABL / "resumo_ablacao.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def statistics(runs):
    """McNemar exato por semente (com x sem), diferença das médias e teste t de Welch."""
    from scipy.stats import binomtest, ttest_ind

    out = {}
    for ds in DSS:
        com, sem = metric(runs, ds, "com", "test_acc"), metric(runs, ds, "sem", "test_acc")
        if not len(com) or not len(sem):
            continue
        o = {"acc_com": com.tolist(), "acc_sem": sem.tolist(),
             "media_com": float(com.mean()), "media_sem": float(sem.mean()),
             "dp_com": float(com.std(ddof=1)) if len(com) > 1 else None,
             "dp_sem": float(sem.std(ddof=1)) if len(sem) > 1 else None,
             "diferenca_pp": float(100 * (com.mean() - sem.mean())),
             "erros_com": float(10000 * (1 - com.mean())), "erros_sem": float(10000 * (1 - sem.mean())),
             "mcnemar": []}
        o["reducao_erros_pct"] = 100 * (1 - o["erros_com"] / o["erros_sem"])
        if len(com) > 1 and len(sem) > 1:
            o["welch_p"] = float(ttest_ind(com, sem, equal_var=False).pvalue)
        for s in SEEDS:
            if (ds, "com", s) in runs and (ds, "sem", s) in runs:
                a, b = runs[(ds, "com", s)]["ok"], runs[(ds, "sem", s)]["ok"]
                n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
                pv = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                o["mcnemar"].append({"semente": s, "so_com_acerta": n01, "so_sem_acerta": n10,
                                     "p_valor": float(pv)})
        for k in ("clean_train_acc", "test_loss", "test_f1_macro"):
            o[f"{k}_com"], o[f"{k}_sem"] = (metric(runs, ds, "com", k).tolist(),
                                           metric(runs, ds, "sem", k).tolist())
        out[ds] = o
    # quanto da queda "sem regularização nenhuma" o aumento de dados explica (CIFAR-10, s42)
    k = [("cifar10", c, 42) for c in ("com", "sem", "noreg")]
    if all(x in runs for x in k):
        c, s, n = (runs[x]["m"]["test_acc"] for x in k)
        out["cifar10_escada_s42"] = {"com": c, "sem_aumento": s, "sem_regularizacao": n,
                                     "fracao_explicada_pelo_aumento": (c - s) / (c - n)}
    (ABL / "estatistica_ablacao.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                  encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Robustez: acurácia no teste transformado
# --------------------------------------------------------------------------- #
def shift(x, dx, dy, mode):
    """Desloca o conteúdo dx px para a direita e dy px para baixo, preenchendo a borda."""
    p = max(abs(dx), abs(dy))
    if p == 0:
        return x
    xp = F.pad(x, (p, p, p, p), mode=mode)
    H, W = x.shape[-2:]
    return xp[..., p - dy:p - dy + H, p - dx:p - dx + W]


def affine(x, deg=0.0, scale=1.0):
    a = math.radians(deg)
    c, s = float(math.cos(a) / scale), float(math.sin(a) / scale)
    theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], device=x.device,
                         dtype=x.dtype).expand(x.shape[0], 2, 3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


@torch.no_grad()
def acc_under(model, x, y, fn, bs=1000):
    ok = 0
    for k in range(0, len(x), bs):
        xb = fn(x[k:k + bs].float().div(255), k)
        with torch.autocast("cuda", dtype=torch.float16):
            ok += (model(xb).argmax(1) == y[k:k + bs]).sum().item()
    return ok / len(x)


def noise_fn(sigma):
    def fn(xb, k):  # mesmo ruído para todos os modelos (gerador fixo por lote)
        g = torch.Generator(device=xb.device).manual_seed(1000 + k)
        return (xb + sigma * torch.randn(xb.shape, device=xb.device, generator=g)).clamp_(0, 1)
    return fn


def shift_curve(model, x, y, ds, max_d):
    mode = "reflect" if ds == "cifar10" else "constant"  # CIFAR: igual ao recorte do treino
    out = []
    for d in range(max_d + 1):
        dirs = [(0, 0)] if d == 0 else [(d, 0), (-d, 0), (0, d), (0, -d)]
        out.append(float(np.mean([acc_under(model, x, y, lambda xb, k, a=a, b=b: shift(xb, a, b, mode))
                                  for a, b in dirs])))
    return {"x": list(range(max_d + 1)), "acc": out}


def robustness(runs):
    res = {}
    for ds in DSS:
        x, y = get_splits(ds)["test"]
        x, y = x.to(DEV), y.to(DEV)
        res[ds] = {}
        for cfg in ("com", "sem"):
            res[ds][cfg] = {}
            for s in seeds_of(runs, ds, cfg):
                model = build_model("minhacnn", ds).to(DEV)
                model.load_state_dict(torch.load(runs[(ds, cfg, s)]["ckpt"], map_location=DEV,
                                                 weights_only=True))
                model.eval()
                r = {"original": acc_under(model, x, y, lambda xb, k: xb)}
                if ds == "cifar10":
                    r["shift"] = shift_curve(model, x, y, ds, 8)
                    r["flip"] = acc_under(model, x, y, lambda xb, k: xb.flip(3))
                    sig = [0, 0.02, 0.04, 0.06, 0.08, 0.10]
                else:
                    r["shift"] = shift_curve(model, x, y, ds, 6)
                    angs = list(range(-45, 46, 5))
                    r["rot"] = {"x": angs, "acc": [acc_under(model, x, y, lambda xb, k, a=a: affine(xb, deg=a))
                                                   for a in angs]}
                    scs = [round(float(v), 2) for v in np.arange(0.6, 1.41, 0.1)]
                    r["scale"] = {"x": scs, "acc": [acc_under(model, x, y, lambda xb, k, c=c: affine(xb, scale=c))
                                                    for c in scs]}
                    sig = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
                r["noise"] = {"x": sig, "acc": [acc_under(model, x, y, noise_fn(v)) for v in sig]}
                res[ds][cfg][str(s)] = r
                print(f"robustez {ds} {cfg} s{s}: original {r['original']:.4f}", flush=True)
                del model
    (ABL / "robustez.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=15, fontweight="bold", color=style.INK)


def fig_examples():
    """O que o componente faz: imagens originais e versões aumentadas."""
    torch.manual_seed(0)
    picks = [("cifar10", 7), ("cifar10", 1), ("mnist", 7), ("mnist", 4)]
    n_aug = 6
    fig, axes = plt.subplots(len(picks), n_aug + 1, figsize=(10.5, 6.6))
    for r, (ds, cls) in enumerate(picks):
        x, y = get_splits(ds)["test"]
        # numpy, não torch: operações do torch na CPU travam neste Windows (OpenMP duplicado)
        i = int(np.flatnonzero(y.numpy() == cls)[0 if ds == "cifar10" else 1])
        img = x[i:i + 1].float().div(255).to(DEV)
        aug = make_augment(ds)(img.repeat(n_aug, 1, 1, 1))
        ims = torch.cat([img, aug]).cpu().numpy()
        for c in range(n_aug + 1):
            ax = axes[r, c]
            im = ims[c]
            ax.imshow(im[0] if im.shape[0] == 1 else im.transpose(1, 2, 0),
                      cmap="gray" if im.shape[0] == 1 else None, interpolation="nearest",
                      vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(c == 0); sp.set_color(style.INK2); sp.set_linewidth(1.6)
            if r == 0:
                ax.set_title("original" if c == 0 else f"aumentada {c}", fontsize=10,
                             color=style.INK2, fontweight="bold" if c == 0 else "normal")
        axes[r, 0].set_ylabel(f"{DS_NAME[ds]}\n{CLASSES[ds][cls]}", fontsize=10.5,
                              color=style.INK2, rotation=0, ha="right", va="center", labelpad=10)
    title(fig, "O componente estudado: aumento de dados")
    fig.tight_layout()
    finish(fig, "ablacao_exemplos.png",
           "CIFAR-10: recorte aleatório com padding por reflexão de 4 px e espelhamento horizontal. "
           "MNIST: rotação de até ±10°, translação de até ±8% e escala de 0,9 a 1,1.\n"
           "Cada época sorteia uma nova versão de cada imagem; validação e teste usam a imagem original.")


def fig_result(runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1.15]})
    for ax, ds in zip(axes, DSS):
        cfgs = [c for c in ("com", "sem", "noreg") if seeds_of(runs, ds, c) or (ds, c, 42) in runs]
        data = []
        for cfg in cfgs:
            ss = seeds_of(runs, ds, cfg) if cfg != "noreg" else [42]
            data.append(np.array([100 * (1 - runs[(ds, cfg, s)]["m"]["test_acc"]) for s in ss]))
        xmax = max(d.max() for d in data)
        for i, (cfg, errs) in enumerate(zip(cfgs, data)):
            lab = CFG_NAME[cfg]
            ax.barh(i, errs.mean(), height=0.62, color=COLORS[lab], edgecolor=style.SURFACE,
                    linewidth=2, zorder=2)
            if len(errs) > 1:  # cada semente como um ponto
                ax.scatter(errs, [i] * len(errs), s=34, color=style.INK, zorder=3,
                           edgecolor=style.SURFACE, linewidth=1.2)
            x0 = max(errs.max(), errs.mean()) + xmax * 0.03
            ax.text(x0, i - 0.08, f"{br(errs.mean())}%" + (f"  ± {br(errs.std(ddof=1))}" if len(errs) > 1 else ""),
                    va="center", fontsize=11, color=style.INK, fontweight="bold")
            n = "3 sementes" if len(errs) == 3 else f"{len(errs)} semente" + ("s" if len(errs) > 1 else "")
            ax.text(x0, i + 0.24, f"{br(100 * errs.mean(), 0)} erros em 10.000 · {n}", va="center",
                    fontsize=8.8, color=style.INK2)
        ax.set_yticks(range(len(cfgs)), [CFG_NAME[c] for c in cfgs], fontsize=10.5)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax * 1.75)
        ax.set_xlabel("erro no teste (%)  ←  menor é melhor")
        ax.set_title(DS_NAME[ds], loc="left", fontsize=14)
        ax.grid(axis="y", visible=False)
    title(fig, "Erro da MinhaCNN no teste com e sem aumento de dados")
    fig.tight_layout()
    finish(fig, "ablacao_resultado.png",
           "Barra = média entre as sementes; pontos pretos = cada semente. Fora o aumento de dados, "
           "tudo é igual: arquitetura, dropout, weight decay, label smoothing, otimizador e épocas.\n"
           "\"Sem nenhuma regularização\" é a ablação da APS anterior (sem aumento, dropout, weight "
           "decay e label smoothing), com uma semente.")


def mean_hist(runs, ds, cfg, key):
    H = [runs[(ds, cfg, s)]["h"] for s in seeds_of(runs, ds, cfg)]
    ep = np.array([e["epoch"] for e in H[0]])
    v = np.array([[e[key] for e in h] for h in H])
    return ep, v.mean(0), v


def fig_curves(runs):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), sharey="row")
    for r, ds in enumerate(DSS):
        cur = {c: (mean_hist(runs, ds, c, "train_acc"), mean_hist(runs, ds, c, "val_acc"))
               for c in ("com", "sem") if seeds_of(runs, ds, c)}
        late = np.concatenate([100 * v[ep >= 3] for pair in cur.values() for (ep, v, _) in pair])
        step = 5 if ds == "cifar10" else 0.5
        lo = np.floor(late.min() / step) * step
        for c, cfg in enumerate(("com", "sem")):
            ax = axes[r, c]
            if cfg not in cur:
                ax.set_visible(False); continue
            (ep, tr, _), (_, va, _) = cur[cfg]
            tr, va = 100 * tr, 100 * va
            lab = CFG_NAME[cfg]; col = COLORS[lab]
            ax.fill_between(ep, tr, va, color=col, alpha=0.13, lw=0)
            ax.plot(ep, tr, color=col, lw=2, alpha=0.5, label="treino")
            ax.plot(ep, va, color=col, lw=2.8, label="validação")
            gap = tr[-1] - va[-1]
            ax.text(0.97, 0.06, f"última época:\ntreino {br(tr[-1], 1)}%  ·  validação {br(va[-1], 1)}%\n"
                                f"diferença: {br(gap, 1)} p.p.",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color=style.INK,
                    bbox=dict(boxstyle="round,pad=0.5", fc=style.SURFACE, ec=style.GRID))
            ax.set_ylim(lo, 100.1 if ds == "mnist" else 101)
            ax.set_yticks([t for t in ax.get_yticks() if lo <= t <= 100])
            ax.set_xlim(0, ep[-1] + 0.5)
            ax.set_xticks(range(0, ep[-1] + 1, 5 if ep[-1] > 20 else 3))
            ax.set_title(f"{DS_NAME[ds]} — {lab.lower()}", loc="left", fontsize=13)
            ax.set_xlabel("época")
            ax.set_ylabel("acurácia (%)" if c == 0 else "")
            ax.legend(loc="center right", frameon=True, facecolor=style.SURFACE,
                      edgecolor=style.GRID, fontsize=10)
    title(fig, "Treino x validação com e sem aumento de dados (média de 3 sementes)")
    fig.tight_layout()
    finish(fig, "ablacao_curvas.png",
           "Treino = acurácia durante a época, com dropout (e aumento de dados, quando ligado). A área "
           "sombreada é a diferença entre treino e validação.\nO eixo vertical começa perto do valor da "
           "época 3 para mostrar o fim do treino; as duas colunas de cada linha usam a mesma escala.")


def fig_val_loss(runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, ds in zip(axes, DSS):
        hi, items = 0, []
        for cfg in ("com", "sem"):
            if not seeds_of(runs, ds, cfg):
                continue
            ep, vl, _ = mean_hist(runs, ds, cfg, "val_loss")
            lab = CFG_NAME[cfg]; col = COLORS[lab]
            ax.plot(ep, vl, color=col, lw=2.6, label=lab)
            items.append((ep[-1] + 0.4, vl[-1], br(vl[-1], 3), style.INK))
            hi = max(hi, vl[ep >= 3].max())
        ax.set_ylim(0, hi * 1.15)
        ax.set_xlim(0, ep[-1] * 1.12)
        spread_labels(ax, items, min_gap=hi * 0.07)
        ax.set_xlabel("época")
        ax.set_ylabel("perda de validação (entropia cruzada)")
        ax.set_title(DS_NAME[ds], loc="left", fontsize=14)
        ax.legend(loc="upper right", frameon=True, facecolor=style.SURFACE, edgecolor=style.GRID, fontsize=10)
    title(fig, "Perda de validação ao longo do treino, com e sem aumento de dados")
    fig.tight_layout()
    finish(fig, "ablacao_perda_validacao.png",
           "Média de 3 sementes. Entropia cruzada sem label smoothing, nas 5.000 imagens de validação; "
           "o número à direita é o valor na última época.\nEixo começa em zero; os valores das primeiras "
           "épocas podem ficar acima do topo.")


def fig_parts(runs):
    need = [("cifar10", c, 42) for c in ("sem", "flip", "crop", "com")]
    if not all(k in runs for k in need):
        print("fig_parts: faltam execuções", [k for k in need if k not in runs]); return
    err = lambda c: 100 * (1 - runs[("cifar10", c, 42)]["m"]["test_acc"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    names = ["nenhum (sem aumento)", "só espelhamento", "só recorte (±4 px)", "recorte + espelhamento"]
    cols = [COLORS[SEM], PARTIAL, PARTIAL, COLORS[COM]]
    vals = [err(c) for c in ("sem", "flip", "crop", "com")]
    a1.barh(range(4), vals, height=0.62, color=cols, edgecolor=style.SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        a1.text(v + max(vals) * 0.02, i, f"{br(v)}%", va="center", fontsize=11, color=style.INK,
                fontweight="bold")
        if i:
            a1.text(v + max(vals) * 0.17, i, f"−{br(vals[0] - v)} p.p.", va="center", fontsize=9.5,
                    color=style.INK2)
    a1.set_yticks(range(4), names, fontsize=10.5)
    a1.invert_yaxis()
    a1.set_xlim(0, max(vals) * 1.4)
    a1.grid(axis="y", visible=False)
    a1.set_xlabel("erro no teste (%)  ←  menor é melhor")
    a1.set_title("Cada parte do aumento sozinha", loc="left", fontsize=13)

    pads = [p for p, c in [(0, "flip"), (2, "pad2"), (4, "com"), (6, "pad6"), (8, "pad8")]
            if ("cifar10", c, 42) in runs]
    cfg_of = {0: "flip", 2: "pad2", 4: "com", 6: "pad6", 8: "pad8"}
    ev = [err(cfg_of[p]) for p in pads]
    a2.axhline(vals[0], color=COLORS[SEM], lw=1.6, ls=(0, (4, 3)))
    a2.text(pads[0], vals[0] + 0.15, f"sem aumento: {br(vals[0])}%", ha="left", va="bottom", fontsize=9.5,
            color=style.INK2)
    a2.plot(pads, ev, color=COLORS[COM], lw=2.4, marker="o", markersize=8,
            markeredgecolor=style.SURFACE, markeredgewidth=1.5, label="recorte + espelhamento")
    for p, v in zip(pads, ev):
        a2.annotate(f"{br(v)}%", (p, v), xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=10, color=style.INK, fontweight="bold" if p == 4 else "normal")
    com = metric(runs, "cifar10", "com", "test_acc")
    if len(com) > 1:  # variação entre sementes no ponto padrão
        sd = 100 * com.std(ddof=1)
        a2.errorbar([4], [err("com")], yerr=[[sd], [sd]], color=style.INK2, capsize=5, lw=1.2, zorder=1)
        a2.annotate(f"padrão (±{br(sd)} entre sementes)", (4, err("com")), xytext=(0, -22),
                    textcoords="offset points", ha="center", fontsize=9, color=style.INK2)
    a2.set_xticks(pads, [f"±{p} px" if p else "0\n(só espelhamento)" for p in pads])
    a2.set_ylim(0, max(vals[0], max(ev)) * 1.25)
    a2.set_xlabel("deslocamento máximo do recorte aleatório")
    a2.set_ylabel("erro no teste (%)")
    a2.set_title("Intensidade do recorte (com espelhamento)", loc="left", fontsize=13)
    title(fig, "CIFAR-10: de onde vem o ganho do aumento de dados")
    fig.tight_layout()
    finish(fig, "ablacao_partes_cifar10.png",
           "Semente 42 em todas as variantes (uma execução por ponto). A barra de erro no ponto padrão "
           "mostra o desvio-padrão entre 3 sementes, para dar a escala do ruído.")


def band(ax, rob, ds, cfg, key, color, label):
    S = rob[ds][cfg]
    xs = np.array(next(iter(S.values()))[key]["x"])
    A = 100 * np.array([S[s][key]["acc"] for s in S])
    ax.fill_between(xs, A.min(0), A.max(0), color=color, alpha=0.18, lw=0)
    ax.plot(xs, A.mean(0), color=color, lw=2.4, label=label, marker="o", markersize=4.5,
            markeredgecolor=style.SURFACE, markeredgewidth=1)
    return xs, A.mean(0)


def fig_robust(rob):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    panels = [
        ("cifar10", "shift", "translação (px, média de 4 direções)", "CIFAR-10 — imagem deslocada"),
        ("cifar10", "flip", None, "CIFAR-10 — imagem espelhada"),
        ("cifar10", "noise", "desvio-padrão do ruído (pixels em [0, 1])", "CIFAR-10 — ruído (controle)"),
        ("mnist", "rot", "rotação (graus)", "MNIST — dígito girado"),
        ("mnist", "shift", "translação (px, média de 4 direções)", "MNIST — dígito deslocado"),
        ("mnist", "scale", "escala (1 = original)", "MNIST — dígito ampliado / reduzido"),
    ]
    for ax, (ds, key, xl, ttl) in zip(axes.ravel(), panels):
        if ds not in rob or not rob[ds].get("com") or not rob[ds].get("sem"):
            ax.set_visible(False); continue
        if key == "flip":  # gráfico de inclinação: original -> espelhada
            for cfg in ("com", "sem"):
                lab = CFG_NAME[cfg]; S = rob[ds][cfg]
                o = 100 * np.mean([S[s]["original"] for s in S])
                f = 100 * np.mean([S[s]["flip"] for s in S])
                ax.plot([0, 1], [o, f], color=COLORS[lab], lw=2.4, marker="o", markersize=9,
                        markeredgecolor=style.SURFACE, markeredgewidth=1.5, label=lab)
                ax.text(-0.06, o, f"{br(o, 1)}%", ha="right", va="center", fontsize=10, color=style.INK)
                ax.text(1.06, f, f"{br(f, 1)}%  ({'−' if f < o else '+'}{br(abs(o - f), 2)} p.p.)",
                        ha="left", va="center", fontsize=10, color=style.INK, fontweight="bold")
            ax.set_xticks([0, 1], ["original", "espelhada"])
            ax.set_xlim(-0.45, 1.7)
            ax.grid(axis="x", visible=False)
        else:
            rng = TRAIN_RANGE.get((ds, key))
            if rng:  # explicada na nota de rodapé (um rótulo aqui colidia com a legenda)
                ax.axvspan(*rng, color=style.GRID, alpha=0.55, lw=0, zorder=0)
            ends = []
            for cfg in ("com", "sem"):
                lab = CFG_NAME[cfg]
                xs, m = band(ax, rob, ds, cfg, key, COLORS[lab], lab)
                ends.append((xs, m))
            # rótulo com a diferença no ponto mais extremo
            xs, mc = ends[0]; _, ms = ends[1]
            j = int(np.argmax(mc - ms)) if key != "noise" else len(xs) - 1
            ax.annotate(f"{'+' if mc[j] >= ms[j] else '−'}{br(abs(mc[j] - ms[j]), 1)} p.p.",
                        (xs[j], (mc[j] + ms[j]) / 2), xytext=(8, 0), textcoords="offset points",
                        fontsize=10, color=style.INK, fontweight="bold", va="center")
            ax.set_xlabel(xl)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, min(hi + (hi - lo) * 0.04, 100.3))
        ax.set_yticks([t for t in ax.get_yticks() if lo <= t <= 100])
        ax.set_ylabel("acurácia no teste (%)")
        ax.set_title(ttl, loc="left", fontsize=12)
        ax.legend(loc="lower center" if key in ("rot", "scale") else "lower left", frameon=True,
                  facecolor=style.SURFACE, edgecolor=style.GRID, fontsize=9)
    title(fig, "O que o aumento de dados ensina: robustez a transformações do teste")
    fig.tight_layout()
    finish(fig, "ablacao_robustez.png",
           "Linha = média de 3 sementes; faixa = mínimo e máximo. A área cinza é a faixa que o aumento "
           "de dados cobre no treino. O ruído é um controle: nenhum dos modelos treinou com ele.\n"
           "Translação no CIFAR-10 com borda refletida (como no recorte do treino); no MNIST, com fundo preto.")


def fig_per_class(runs, ds="cifar10"):
    if not seeds_of(runs, ds, "com") or not seeds_of(runs, ds, "sem"):
        print("fig_per_class: faltam execuções"); return
    cls = CLASSES[ds]
    pc = lambda cfg: 100 * np.mean([runs[(ds, cfg, s)]["m"]["per_class_acc"] for s in seeds_of(runs, ds, cfg)], 0)
    vc, vs = pc("com"), pc("sem")
    order = np.argsort(vc)[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(cls) + 1.6))
    y = np.arange(len(cls))
    for yi, k in zip(y, order):
        ax.plot([vs[k], vc[k]], [yi, yi], color=style.AXIS, lw=3, zorder=1, solid_capstyle="round")
    for v, lab in ((vs, SEM), (vc, COM)):
        ax.scatter(v[order], y, s=110, color=COLORS[lab], zorder=3, label=lab,
                   edgecolor=style.SURFACE, linewidth=1.5)
    for yi, k in zip(y, order):
        d = vc[k] - vs[k]
        ax.text(100.4, yi, f"{'+' if d >= 0 else '−'}{br(abs(d), 1)} p.p.", va="center", fontsize=10,
                color=style.INK if abs(d) >= 3 else style.INK2, fontweight="bold" if abs(d) >= 3 else "normal")
    ax.text(100.4, -0.9, "ganho", fontsize=9, color=style.MUTED, va="center")
    ax.set_yticks(y, [cls[k] for k in order], fontsize=11)
    ax.invert_yaxis()
    lo = np.floor(min(vc.min(), vs.min()) / 2) * 2 - 1
    ax.set_xlim(lo, 103.4)
    ax.set_xticks(np.arange(np.ceil(lo / 2) * 2, 100.1, 2))
    ax.set_xlabel("acurácia no teste (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", frameon=True, facecolor=style.SURFACE, edgecolor=style.GRID, fontsize=10)
    ax.set_title(f"Acurácia por classe com e sem aumento de dados — {DS_NAME[ds]}", loc="left",
                 fontsize=15, fontweight="bold", color=style.INK)
    fig.tight_layout()
    finish(fig, f"ablacao_por_classe_{ds}.png",
           "Média de 3 sementes. Classes ordenadas pela acurácia com aumento de dados; à direita, quanto "
           "o aumento de dados acrescenta em cada classe (p.p. = pontos percentuais).")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-robust", action="store_true")
    a = ap.parse_args(argv)
    style.setup()
    ABL.mkdir(parents=True, exist_ok=True)
    runs = load_all()
    print("execuções:", sorted(runs), flush=True)
    summary(runs)
    st = statistics(runs)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
                      for k, v in st.items()}, indent=1, ensure_ascii=False), flush=True)
    rob_file = ABL / "robustez.json"
    rob = json.loads(rob_file.read_text(encoding="utf-8")) if a.no_robust and rob_file.exists() \
        else robustness(runs)
    fig_examples()
    fig_result(runs)
    fig_curves(runs)
    fig_val_loss(runs)
    fig_parts(runs)
    fig_robust(rob)
    fig_per_class(runs)
    print("figuras salvas em figures/ablacao_*.png", flush=True)


if __name__ == "__main__":
    main()
