"""Gráficos e tabelas de comparação a partir de results/<tag>/ (seaborn + matplotlib).

Princípios: escalas lineares, eixo y igual entre painéis comparados, um assunto por
painel, rótulo com o valor final em cada curva e legendas que explicam a leitura.

Uso:  python -m src.plots
"""
from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import PowerNorm

from . import style
from .data import CLASSES, ROOT, get_splits

FIG = ROOT / "figures"
RES = ROOT / "results"

LABELS = {  # prefixo do tag -> nome exibido (ordem fixa = ordem da paleta)
    "minhacnn": "MinhaCNN",
    "resnet18_ft": "ResNet18 (fine-tuning)",
    "resnet18_frozen": "ResNet18 (extração)",
    "resnet18_scratch": "ResNet18 do zero",
    "minhacnn_noreg": "MinhaCNN sem regularização",
}
DS_NAME = {"mnist": "MNIST", "cifar10": "CIFAR-10"}
FAMILIES = [
    ("CNN própria, treinada do zero", ["MinhaCNN", "MinhaCNN sem regularização"]),
    ("ResNet18", ["ResNet18 (fine-tuning)", "ResNet18 (extração)", "ResNet18 do zero"]),
]


def br(x, d=2):
    """Número com vírgula decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def load_all():
    runs = {}
    for d in sorted(RES.iterdir()):
        if not (d / "metrics.json").exists():
            continue
        raw = (d / "metrics.json").read_bytes()
        try:
            m = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:  # gravado com a codificação padrão do Windows
            m = json.loads(raw.decode("cp1252"))
        h = json.loads((d / "history.json").read_text(encoding="utf-8"))
        prefix = d.name.rsplit("_", 1)[0]
        runs[(LABELS[prefix], m["dataset"])] = {"m": m, "h": h, "tag": d.name}
    return runs


def ordered(runs, ds):
    return [lab for lab in LABELS.values() if (lab, ds) in runs]


def curves(runs, lab, ds):
    h = runs[(lab, ds)]["h"]
    ep = np.array([e["epoch"] for e in h])
    tr = 100 * np.array([e["train_acc"] for e in h])
    va = 100 * np.array([e["val_acc"] for e in h])
    return ep, tr, va


def spread_labels(ax, items, min_gap):
    """Escreve rótulos à direita do fim das curvas sem sobreposição.
    items: lista de (x, y, texto, cor) em coordenadas de dados; min_gap em unidades de y."""
    items = sorted(items, key=lambda t: t[1])
    groups = [[i] for i in range(len(items))]
    pos = [it[1] for it in items]
    changed = True
    while changed:  # junta rótulos que colidem e centraliza o grupo no valor médio
        changed = False
        for g in range(len(groups) - 1):
            if pos[groups[g + 1][0]] - pos[groups[g][-1]] < min_gap:
                merged = groups[g] + groups[g + 1]
                center = np.mean([items[i][1] for i in merged])
                start = center - min_gap * (len(merged) - 1) / 2
                for k, i in enumerate(merged):
                    pos[i] = start + k * min_gap
                groups[g:g + 2] = [merged]
                changed = True
                break
    ys = pos
    top = ax.get_ylim()[1]
    if ys and ys[-1] > top - min_gap * 0.3:
        shift = ys[-1] - (top - min_gap * 0.3)
        ys = [y - shift for y in ys]
    for (x, y, t, c), yl in zip(items, ys):
        ax.annotate(t, (x, y), xytext=(x, yl), textcoords="data", va="center", ha="left",
                    fontsize=10, color=c, fontweight="bold",
                    xycoords="data", annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.15", fc=style.SURFACE, ec="none"))


def finish(fig, fname, note=None):
    if note:
        fig.text(0.01, -0.01, note, fontsize=9, color=style.MUTED, ha="left", va="top")
    fig.savefig(FIG / fname)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 1) Erro no teste
# --------------------------------------------------------------------------- #
def fig_test_error(runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [3, 5]})
    for ax, ds in zip(axes, ["mnist", "cifar10"]):
        labs = ordered(runs, ds)
        df = pd.DataFrame({"modelo": labs,
                           "erro": [100 * (1 - runs[(l, ds)]["m"]["test_acc"]) for l in labs]})
        sns.barplot(data=df, y="modelo", x="erro", hue="modelo", palette=style.COLORS,
                    legend=False, ax=ax, width=0.65, edgecolor=style.SURFACE, linewidth=2,
                    saturation=1)
        for i, e in enumerate(df["erro"]):
            ax.text(e + df["erro"].max() * 0.02, i, f"{br(e)}%", va="center", fontsize=11,
                    color=style.INK, fontweight="bold")
            ax.text(e + df["erro"].max() * 0.02, i + 0.3, f"acerta {br(100 - e)}%", va="center",
                    fontsize=8.5, color=style.INK2)
        ax.set_xlim(0, df["erro"].max() * 1.35)
        ax.set_xlabel("erro no teste (%)  ←  menor é melhor")
        ax.set_ylabel("")
        ax.set_title(DS_NAME[ds], loc="left", fontsize=14)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Quanto cada modelo erra no conjunto de teste (10.000 imagens)", x=0.01,
                 ha="left", fontsize=15, fontweight="bold", color=style.INK)
    fig.tight_layout()
    finish(fig, "comparacao_erro_teste.png")


# --------------------------------------------------------------------------- #
# 2) Curvas de validação: linhas = dataset, colunas = família de modelos (mesmo eixo y)
# --------------------------------------------------------------------------- #
def fig_val_curves(runs):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharey="row",
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for r, ds in enumerate(["mnist", "cifar10"]):
        row_min = min(curves(runs, l, ds)[2].min() for l in ordered(runs, ds))
        lo = np.floor(row_min / 5) * 5 if ds == "cifar10" else np.floor(row_min) - 1
        best_minha = curves(runs, "MinhaCNN", ds)[2].max()
        for c, (fam, labs) in enumerate(FAMILIES):
            ax = axes[r, c]
            labs = [l for l in labs if (l, ds) in runs]
            items, max_ep = [], 1
            for lab in labs:
                ep, _, va = curves(runs, lab, ds)
                max_ep = max(max_ep, ep[-1])
                ax.plot(ep, va, color=style.COLORS[lab], lw=2.6, label=lab,
                        marker="o" if len(ep) <= 6 else None, markersize=7,
                        markeredgecolor=style.SURFACE, markeredgewidth=1.5)
                items.append((ep[-1] + max(0.15, ep[-1] * 0.025), va[-1], f"{br(va[-1], 1)}%",
                              style.COLORS[lab]))
            if c == 1 and ds == "cifar10":  # referência: melhor validação da MinhaCNN
                ax.axhline(best_minha, color=style.INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
                ax.text(0.98, best_minha + 0.4,
                        f"melhor validação da MinhaCNN: {br(best_minha, 1)}%",
                        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                        fontsize=9, color=style.INK2)
            ax.set_ylim(lo, 100.5)
            ax.set_xlim(0.5 if max_ep <= 6 else 0, max_ep * (1.22 if max_ep > 6 else 1.3))
            ax.set_xticks(range(1, max_ep + 1) if max_ep <= 6 else range(0, max_ep + 1, 5))
            spread_labels(ax, items, min_gap=(100.5 - lo) * 0.06)
            ax.set_title(f"{DS_NAME[ds]} — {fam}", loc="left", fontsize=13)
            ax.set_xlabel("época")
            ax.set_ylabel("acurácia na validação (%)" if c == 0 else "")
            ax.legend(loc="lower right", frameon=True, facecolor=style.SURFACE,
                      edgecolor=style.GRID, fontsize=9.5)
    fig.suptitle("Acurácia na validação ao longo do treino", x=0.01, ha="left", fontsize=15,
                 fontweight="bold", color=style.INK)
    fig.tight_layout()
    finish(fig, "curvas_validacao.png",
           "Cada linha do gráfico é um dataset e usa o mesmo eixo vertical nos dois painéis. "
           "No CIFAR-10, a linha tracejada repete no painel da ResNet18 o melhor valor da MinhaCNN.")


# --------------------------------------------------------------------------- #
# 3) Treino x validação, com o gap sombreado
# --------------------------------------------------------------------------- #
def fig_train_val(runs, ds, labs, fname, title):
    labs = [l for l in labs if (l, ds) in runs]
    fig, axes = plt.subplots(1, len(labs), figsize=(6.6 * len(labs), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    allv = np.concatenate([np.concatenate(curves(runs, l, ds)[1:]) for l in labs])
    lo = np.floor(allv.min() / 5) * 5 if ds == "cifar10" else np.floor(allv.min()) - 1
    for ax, lab in zip(axes, labs):
        ep, tr, va = curves(runs, lab, ds)
        col = style.COLORS[lab]
        ax.fill_between(ep, tr, va, color=col, alpha=0.13, lw=0)
        mk = dict(marker="o", markersize=6, markeredgecolor=style.SURFACE) if len(ep) <= 6 else {}
        ax.plot(ep, tr, color=col, lw=2, alpha=0.5, label="treino", **mk)
        ax.plot(ep, va, color=col, lw=2.8, label="validação", **mk)
        gap = tr[-1] - va[-1]
        ax.text(0.97, 0.06,
                f"última época:\ntreino {br(tr[-1], 1)}%  ·  validação {br(va[-1], 1)}%\n"
                f"diferença: {br(gap, 1)} p.p.",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color=style.INK,
                bbox=dict(boxstyle="round,pad=0.5", fc=style.SURFACE, ec=style.GRID))
        ax.set_title(lab, loc="left", fontsize=13, color=col)
        ax.set_xlabel("época")
        ax.set_ylim(lo, 101)
        n = ep[-1]
        ax.set_xticks(range(1, n + 1) if n <= 6 else range(0, n + 1, 5))
        ax.legend(loc="center right", frameon=True, facecolor=style.SURFACE,
                  edgecolor=style.GRID, fontsize=10)
    axes[0].set_ylabel("acurácia (%)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold", color=style.INK)
    fig.tight_layout()
    finish(fig, fname, "Treino = acurácia durante a época, com aumento de dados e dropout "
                       "ativos (quando o modelo os usa). A área sombreada é a diferença entre "
                       "treino e validação.")


# --------------------------------------------------------------------------- #
# 4) Matrizes de confusão (seaborn heatmap)
# --------------------------------------------------------------------------- #
def fig_confusion(runs, ds, labs):
    cls = CLASSES[ds]
    fig, axes = plt.subplots(1, len(labs), figsize=(7.2 * len(labs), 6.6))
    axes = np.atleast_1d(axes)
    for ax, lab in zip(axes, labs):
        cm = np.array(runs[(lab, ds)]["m"]["confusion_matrix"], dtype=float)
        pct = 100 * cm / cm.sum(1, keepdims=True)
        if ds == "cifar10":
            annot = np.where(pct >= 1, np.vectorize(lambda v: br(v, 0 if v >= 10 else 1))(pct), "")
        else:
            annot = np.where(cm > 0, cm.astype(int).astype(str), "")
        sns.heatmap(pct, annot=annot, fmt="", cmap=style.SEQ, norm=PowerNorm(0.45, 0, 100),
                    cbar=False, square=True, linewidths=1, linecolor=style.SURFACE,
                    xticklabels=cls, yticklabels=cls, ax=ax, annot_kws={"fontsize": 9.5})
        ax.set_xlabel("classe prevista pelo modelo", fontsize=11)
        ax.set_ylabel("classe real", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        for t in ax.get_xticklabels():
            t.set_ha("right")
        acc = runs[(lab, ds)]["m"]["test_acc"]
        ax.set_title(f"{lab}  ·  acurácia {br(100 * acc)}%", loc="left", fontsize=13,
                     color=style.COLORS[lab])
    unit = ("cada linha soma 100%: o número é a % das imagens daquela classe real"
            if ds == "cifar10" else "números = quantidade de imagens de teste")
    fig.suptitle(f"Matriz de confusão — {DS_NAME[ds]}", x=0.01, ha="left", fontsize=15,
                 fontweight="bold", color=style.INK)
    fig.tight_layout()
    finish(fig, f"matriz_confusao_{ds}.png",
           f"Diagonal = acertos; fora da diagonal = confusões ({unit}).")


# --------------------------------------------------------------------------- #
# 5) Acurácia por classe (gráfico de halteres: um ponto por modelo)
# --------------------------------------------------------------------------- #
def fig_per_class(runs, ds, a="MinhaCNN", b="ResNet18 (fine-tuning)"):
    cls = CLASSES[ds]
    va = 100 * np.array(runs[(a, ds)]["m"]["per_class_acc"])
    vb = 100 * np.array(runs[(b, ds)]["m"]["per_class_acc"])
    order = np.argsort(va)[::-1]           # mais fácil no topo, mais difícil embaixo
    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(cls) + 1.6))
    y = np.arange(len(cls))
    for yi, k in zip(y, order):
        ax.plot([va[k], vb[k]], [yi, yi], color=style.AXIS, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(va[order], y, s=110, color=style.COLORS[a], zorder=3, label=a,
               edgecolor=style.SURFACE, linewidth=1.5)
    ax.scatter(vb[order], y, s=110, color=style.COLORS[b], zorder=3, label=b,
               edgecolor=style.SURFACE, linewidth=1.5)
    xmax = 100.0
    for yi, k in zip(y, order):
        d = vb[k] - va[k]
        ax.text(xmax + 0.4, yi, f"{'+' if d >= 0 else '−'}{br(abs(d), 1)} p.p.", va="center",
                fontsize=10, color=style.INK if abs(d) >= 3 else style.INK2,
                fontweight="bold" if abs(d) >= 3 else "normal")
    ax.text(xmax + 0.4, -0.9, "diferença", fontsize=9, color=style.MUTED, va="center")
    ax.set_yticks(y, [cls[k] for k in order], fontsize=11)
    ax.invert_yaxis()
    lo = np.floor(min(va.min(), vb.min()) / 2) * 2 - 1
    ax.set_xlim(lo, xmax + 3.2)
    ax.set_xticks([t for t in np.arange(np.ceil(lo / 2) * 2, xmax + 0.1, 2)])
    ax.set_xlabel("acurácia no teste (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", frameon=True, facecolor=style.SURFACE, edgecolor=style.GRID,
              fontsize=10)
    ax.set_title(f"Acurácia por classe — {DS_NAME[ds]}", loc="left", fontsize=15,
                 fontweight="bold", color=style.INK)
    fig.tight_layout()
    finish(fig, f"acuracia_por_classe_{ds}.png",
           f"Cada linha é uma classe; a barra cinza liga os dois modelos. "
           f"À direita, quanto a {b} acerta a mais (p.p. = pontos percentuais).")


# --------------------------------------------------------------------------- #
# 6) Custo x desempenho
# --------------------------------------------------------------------------- #
def fig_cost(runs, ds="cifar10"):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for lab in ordered(runs, ds):
        m = runs[(lab, ds)]["m"]
        t, a = m["train_time_s"] / 60, 100 * m["test_acc"]
        ax.scatter([t], [a], s=180, color=style.COLORS[lab], edgecolor=style.SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(f"{lab}: {br(a)}%\n{br(m['params'] / 1e6, 1)} M parâmetros · "
                    + f"{m['inference_img_per_s']:,.0f}".replace(",", ".") + " img/s",
                    (t, a), xytext=(12, 0), textcoords="offset points", fontsize=9.5,
                    color=style.INK2, va="center")
    ax.set_xlabel("tempo total de treino (minutos, RTX 3050 Laptop)")
    ax.set_ylabel("acurácia no teste (%)")
    ax.set_xlim(0, ax.get_xlim()[1] * 1.55)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.3, hi + 0.5)
    ax.set_title(f"Custo x desempenho — {DS_NAME[ds]}", loc="left", fontsize=15,
                 fontweight="bold", color=style.INK)
    fig.tight_layout()
    finish(fig, f"custo_desempenho_{ds}.png",
           "img/s = imagens classificadas por segundo na inferência (quanto maior, mais rápido).")


# --------------------------------------------------------------------------- #
# 7) Exemplos de erros
# --------------------------------------------------------------------------- #
def fig_errors(runs, ds, lab="MinhaCNN", n=24):
    tag = runs[(lab, ds)]["tag"]
    p = np.load(RES / tag / "test_preds.npz")
    x, _ = get_splits(ds)["test"]
    wrong_all = np.where(p["y_true"] != p["y_pred"])[0]
    conf = p["probs"].astype(np.float32).max(1)
    wrong = wrong_all[np.argsort(-conf[wrong_all])][:n]   # erros mais "confiantes"
    cols = 8; rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 2.15 * rows))
    cls = CLASSES[ds]
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        img = x[i].numpy()
        ax.imshow(img[0] if img.shape[0] == 1 else img.transpose(1, 2, 0),
                  cmap="gray" if img.shape[0] == 1 else None, interpolation="nearest")
        ax.set_title(f"real: {cls[p['y_true'][i]]}\nprevisto: {cls[p['y_pred'][i]]} "
                     f"({br(100 * conf[i], 0)}%)", fontsize=8, color=style.INK2)
    fig.suptitle(f"{lab}: os erros com maior confiança no teste do {DS_NAME[ds]} "
                 f"({len(wrong_all)} erros em {len(p['y_true'])} imagens)",
                 fontsize=13, fontweight="bold", color=style.INK)
    fig.tight_layout()
    fig.savefig(FIG / f"erros_{ds}.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Tabelas e estatística
# --------------------------------------------------------------------------- #
def summary_table(runs):
    rows = []
    for ds in ["mnist", "cifar10"]:
        for lab in ordered(runs, ds):
            m = runs[(lab, ds)]["m"]
            rows.append({
                "dataset": DS_NAME[ds], "modelo": lab, "parametros": m["params"],
                "parametros_treinaveis": m["trainable_params"],
                "epocas": m["config"]["epochs"], "melhor_epoca": m["best_epoch"],
                "acc_teste": m["test_acc"], "erro_teste": 1 - m["test_acc"],
                "f1_macro": m["test_f1_macro"], "precisao_macro": m["test_precision_macro"],
                "recall_macro": m["test_recall_macro"], "perda_teste": m["test_loss"],
                "acc_treino_limpo": m["clean_train_acc"],
                "gap_treino_teste": m["clean_train_acc"] - m["test_acc"],
                "tempo_treino_min": m["train_time_s"] / 60,
                "tempo_epoca_s": m["mean_epoch_time_s"],
                "inferencia_img_s": m["inference_img_per_s"],
            })
    with open(RES / "resumo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["| Dataset | Modelo | Parâmetros | Épocas | Acurácia teste | F1 macro | "
          "Acc. treino | Gap | Tempo treino | Inferência |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['dataset']} | {r['modelo']} | {r['parametros'] / 1e6:.2f} M | {r['epocas']} | "
                  f"{100 * r['acc_teste']:.2f}% | {r['f1_macro']:.4f} | "
                  f"{100 * r['acc_treino_limpo']:.2f}% | {100 * r['gap_treino_teste']:.2f} pp | "
                  f"{r['tempo_treino_min']:.1f} min | {r['inferencia_img_s']:,.0f} img/s |")
    (RES / "resumo.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def stats(runs):
    """IC de 95% (Wilson) da acurácia e teste de McNemar exato entre pares de modelos."""
    from scipy.stats import binomtest

    def wilson(k, n, z=1.96):
        ph = k / n; den = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / den
        h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        return c - h, c + h

    out = {"ic95": {}, "mcnemar": []}
    for ds in ["mnist", "cifar10"]:
        preds = {}
        for lab in ordered(runs, ds):
            p = np.load(RES / runs[(lab, ds)]["tag"] / "test_preds.npz")
            ok = p["y_true"] == p["y_pred"]
            preds[lab] = ok
            lo, hi = wilson(ok.sum(), len(ok))
            out["ic95"][f"{DS_NAME[ds]} | {lab}"] = [float(lo), float(hi)]
        labs = list(preds)
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                a, b = preds[labs[i]], preds[labs[j]]
                b01 = int((a & ~b).sum()); b10 = int((~a & b).sum())
                pv = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
                out["mcnemar"].append({"dataset": DS_NAME[ds], "a": labs[i], "b": labs[j],
                                       "so_a_acerta": b01, "so_b_acerta": b10, "p_valor": float(pv)})
    (RES / "estatistica.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    return out


def main():
    style.setup()
    FIG.mkdir(exist_ok=True)
    runs = load_all()
    print("execuções encontradas:", sorted(runs))
    summary_table(runs)
    stats(runs)
    fig_test_error(runs)
    fig_val_curves(runs)
    fig_train_val(runs, "cifar10", ["MinhaCNN", "MinhaCNN sem regularização"],
                  "regularizacao_cifar10.png",
                  "Efeito da regularização na MinhaCNN (CIFAR-10)")
    fig_train_val(runs, "cifar10", ["MinhaCNN", "ResNet18 (fine-tuning)"],
                  "treino_validacao_cifar10.png", "Treino x validação — CIFAR-10")
    fig_train_val(runs, "mnist", ["MinhaCNN", "ResNet18 (fine-tuning)"],
                  "treino_validacao_mnist.png", "Treino x validação — MNIST")
    main_labs = ["MinhaCNN", "ResNet18 (fine-tuning)"]
    for ds in ["mnist", "cifar10"]:
        fig_confusion(runs, ds, [l for l in main_labs if (l, ds) in runs])
        if all((l, ds) in runs for l in main_labs):
            fig_per_class(runs, ds)
        fig_errors(runs, ds)
    fig_cost(runs, "cifar10")
    print("figuras salvas em", FIG)


if __name__ == "__main__":
    main()
