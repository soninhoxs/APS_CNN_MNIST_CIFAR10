"""Gráficos e tabelas de comparação a partir de results/<tag>/.

Uso:  python -m src.plots
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

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


def end_label(ax, x, y, text, color=style.INK2, dx=0.6):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8, color=color)


def place_end_labels(ax, items, min_gap=0.055):
    """Rótulos no fim das linhas, afastados verticalmente para não se sobreporem.
    items: lista de (x, y, texto, cor) em coordenadas de dados."""
    to_ax = ax.transAxes.inverted()
    pts = []
    for x, y, t, c in items:
        fx, fy = to_ax.transform(ax.transData.transform((x, y)))
        pts.append([fy, fx, t, c])
    pts.sort(key=lambda p: p[0])
    for i in range(1, len(pts)):
        if pts[i][0] - pts[i - 1][0] < min_gap:
            pts[i][0] = pts[i - 1][0] + min_gap
    over = pts[-1][0] - 0.98 if pts else 0
    if over > 0:
        for p in pts:
            p[0] -= over
    for fy, fx, t, c in pts:
        ax.text(fx + 0.012, fy, t, transform=ax.transAxes, va="center", fontsize=8, color=c)


def epoch_axis(ax, max_ep, extra=1.5):
    """Espaço à direita para rótulos, mas eixo/ticks só até a última época."""
    step = 5 if max_ep > 10 else 1
    ax.set_xlim(0 if max_ep > 10 else 0.5, max_ep * extra)
    ax.set_xticks([t for t in range(0, max_ep + 1, step) if t >= (0 if max_ep > 10 else 1)])
    ax.spines["bottom"].set_bounds(ax.get_xticks()[0], max_ep)


# --------------------------------------------------------------------------- #
def fig_test_error(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, ds in zip(axes, ["mnist", "cifar10"]):
        labs = ordered(runs, ds)
        err = [100 * (1 - runs[(l, ds)]["m"]["test_acc"]) for l in labs]
        y = np.arange(len(labs))[::-1]
        ax.barh(y, err, height=0.62, color=[style.COLORS[l] for l in labs],
                edgecolor=style.SURFACE, linewidth=2)
        for yi, e, l in zip(y, err, labs):
            acc = 100 - e
            ax.annotate(f"{br(e)}%  (acurácia {br(acc)}%)", (e, yi), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=style.INK2)
        ax.set_yticks(y, labs)
        ax.set_xlim(0, max(err) * 1.7)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("erro no conjunto de teste (%) — menor é melhor")
        ax.set_title(f"{DS_NAME[ds]}", loc="left")
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Erro de teste por modelo", fontsize=12, fontweight="bold", x=0.01,
                 ha="left", color=style.INK)
    fig.tight_layout()
    fig.savefig(FIG / "comparacao_erro_teste.png"); plt.close(fig)


def fig_val_curves(runs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for r, ds in enumerate(["mnist", "cifar10"]):
        for c, (key, ylab) in enumerate([("val_acc", "acurácia na validação (%)"),
                                         ("val_loss", "perda na validação (entropia cruzada)")]):
            ax = axes[r, c]
            for lab in ordered(runs, ds):
                h = runs[(lab, ds)]["h"]
                ep = [e["epoch"] for e in h]
                v = [100 * e[key] if key == "val_acc" else e[key] for e in h]
                ax.plot(ep, v, color=style.COLORS[lab], label=lab, marker="o" if len(ep) <= 5 else None,
                        markersize=4)
            ax.set_xlabel("época"); ax.set_ylabel(ylab)
            ax.set_title(f"{DS_NAME[ds]}", loc="left")
            if key == "val_acc":
                lo = 96 if ds == "mnist" else 40
                ax.set_ylim(lo, 100)
            else:
                ax.set_yscale("log")
    h, l = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    seen = dict(zip(l, h)); seen.update(dict(zip(l2, h2)))
    order = [x for x in LABELS.values() if x in seen]
    fig.legend([seen[x] for x in order], order, loc="upper center", ncol=len(order),
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Curvas de validação por época", fontsize=12, fontweight="bold",
                 y=1.06, color=style.INK)
    fig.tight_layout()
    fig.savefig(FIG / "curvas_validacao.png"); plt.close(fig)


def fig_train_vs_val(runs, ds, labs, fname, title, split=False):
    present = [l for l in labs if (l, ds) in runs]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present) if len(present) > 1 and split else 2,
                             figsize=(11, 3.9))
    panels = ([(ax, "acc", [l]) for ax, l in zip(axes, present)] if split else
              [(axes[0], "acc", present), (axes[1], "loss", present)])
    for ax, k, plabs in panels:
        ylab = "acurácia (%)" if k == "acc" else "perda (entropia cruzada)"
        items, max_ep = [], 1
        for lab in plabs:
            h = runs[(lab, ds)]["h"]
            ep = [e["epoch"] for e in h]
            max_ep = max(max_ep, ep[-1])
            f = (lambda v: 100 * v) if k == "acc" else (lambda v: v)
            tr = [f(e[f"train_{k}"]) for e in h]; va = [f(e[f"val_{k}"]) for e in h]
            col = style.COLORS[lab]
            mk = "o" if len(ep) <= 5 else None
            ax.plot(ep, tr, color=col, linewidth=1.2, alpha=0.55, marker=mk, markersize=3)
            ax.plot(ep, va, color=col, linewidth=2.2, marker=mk, markersize=4)
            short = "" if split else f"{lab}: "
            items += [(ep[-1], tr[-1], f"{short}treino", col), (ep[-1], va[-1], f"{short}validação", col)]
        ax.set_xlabel("época"); ax.set_ylabel(ylab)
        if k == "loss":
            ax.set_yscale("log")
        if split:
            ax.set_title(plabs[0], loc="left")
        epoch_axis(ax, max_ep, 1.35 if split else 1.55)
        fig.canvas.draw()
        place_end_labels(ax, items)
    fig.suptitle(title, fontsize=12, fontweight="bold", color=style.INK, x=0.01, ha="left")
    fig.text(0.01, -0.03, "Linha fina = treino (com aumento de dados/dropout ativos quando houver);"
             " linha grossa = validação.", fontsize=8, color=style.MUTED)
    fig.tight_layout()
    fig.savefig(FIG / fname); plt.close(fig)


def fig_confusion(runs, ds, labs):
    fig, axes = plt.subplots(1, len(labs), figsize=(6.2 * len(labs), 5.6))
    cls = CLASSES[ds]
    for ax, lab in zip(np.atleast_1d(axes), labs):
        cm = np.array(runs[(lab, ds)]["m"]["confusion_matrix"], dtype=float)
        cmn = 100 * cm / cm.sum(1, keepdims=True)
        vmax = 100 if ds == "cifar10" else 100
        ax.imshow(np.sqrt(cmn), cmap=style.SEQ, vmin=0, vmax=np.sqrt(vmax))
        for i in range(10):
            for j in range(10):
                v = cmn[i, j]
                if (ds == "cifar10" and v >= 1) or (ds == "mnist" and cm[i, j] > 0):
                    txt = br(v, 0 if v >= 10 else 1) if ds == "cifar10" else str(int(cm[i, j]))
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color="white" if v > 40 else style.INK2)
        ax.set_xticks(range(10), cls, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(10), cls, fontsize=8)
        ax.set_xlabel("classe prevista"); ax.set_ylabel("classe real")
        ax.grid(False)
        acc = runs[(lab, ds)]["m"]["test_acc"]
        ax.set_title(f"{lab} — acurácia {br(100 * acc)}%", loc="left")
    unit = "valores em % de cada classe real" if ds == "cifar10" else "valores = nº de imagens"
    fig.suptitle(f"Matrizes de confusão no teste — {DS_NAME[ds]} ({unit})" if len(labs) > 1 else
                 f"Matriz de confusão no teste — {DS_NAME[ds]} ({unit})", fontsize=12,
                 fontweight="bold", color=style.INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / f"matriz_confusao_{ds}.png"); plt.close(fig)


def fig_per_class(runs, ds, labs):
    cls = CLASSES[ds]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    n = len(labs); w = 0.8 / n
    x = np.arange(10)
    for k, lab in enumerate(labs):
        v = 100 * np.array(runs[(lab, ds)]["m"]["per_class_acc"])
        ax.bar(x + (k - (n - 1) / 2) * w, v, width=w, color=style.COLORS[lab], label=lab,
               edgecolor=style.SURFACE, linewidth=2)
    lo = 70 if ds == "cifar10" else 97
    ax.set_ylim(lo, 100)
    ax.set_xticks(x, cls)
    ax.set_ylabel("acurácia no teste (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", ncol=n)
    m0 = 100 * np.array(runs[(labs[0], ds)]["m"]["per_class_acc"])
    worst = int(m0.argmin())
    ax.annotate(f"pior classe da {labs[0]}: {cls[worst]} ({br(m0[worst], 1)}%)",
                (worst, m0[worst]), xytext=(0, -22), textcoords="offset points",
                ha="center", fontsize=8, color=style.INK2)
    ax.set_title(f"Acurácia por classe — {DS_NAME[ds]} (eixo começa em {lo}%)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / f"acuracia_por_classe_{ds}.png"); plt.close(fig)


def fig_cost(runs, ds="cifar10"):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for lab in ordered(runs, ds):
        m = runs[(lab, ds)]["m"]
        t = m["train_time_s"] / 60
        a = 100 * m["test_acc"]
        ax.scatter([t], [a], s=90, color=style.COLORS[lab], edgecolor=style.SURFACE, linewidth=2,
                   zorder=3)
        ax.annotate(f"{lab}\n{m['params'] / 1e6:.1f} M parâmetros".replace(".", ","), (t, a),
                    xytext=(8, -2), textcoords="offset points", fontsize=8, color=style.INK2,
                    va="center")
    ax.set_xlabel("tempo de treino (min, RTX 3050 Laptop)")
    ax.set_ylabel("acurácia no teste (%)")
    ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.35)
    ax.set_title(f"Custo x desempenho — {DS_NAME[ds]}", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / f"custo_desempenho_{ds}.png"); plt.close(fig)


def fig_errors(runs, ds, lab="MinhaCNN", n=24):
    tag = runs[(lab, ds)]["tag"]
    p = np.load(RES / tag / "test_preds.npz")
    x, _ = get_splits(ds)["test"]
    wrong = np.where(p["y_true"] != p["y_pred"])[0]
    conf = p["probs"].astype(np.float32).max(1)
    wrong = wrong[np.argsort(-conf[wrong])][:n]          # erros mais "confiantes"
    cols = 8; rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.45 * cols, 2.0 * rows))
    cls = CLASSES[ds]
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        img = x[i].numpy()
        ax.imshow(img[0] if img.shape[0] == 1 else img.transpose(1, 2, 0),
                  cmap="gray" if img.shape[0] == 1 else None, interpolation="nearest")
        ax.set_title(f"real: {cls[p['y_true'][i]]}\nprev.: {cls[p['y_pred'][i]]} "
                     f"({br(100 * conf[i], 0)}%)", fontsize=7, color=style.INK2)
    fig.suptitle(f"{lab} — erros mais confiantes no teste do {DS_NAME[ds]} "
                 f"({len(np.where(p['y_true'] != p['y_pred'])[0])} erros em {len(p['y_true'])})",
                 fontsize=11, fontweight="bold", color=style.INK)
    fig.tight_layout()
    fig.savefig(FIG / f"erros_{ds}.png"); plt.close(fig)


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
    import csv
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
    fig_train_vs_val(runs, "cifar10", ["MinhaCNN", "MinhaCNN sem regularização"],
                     "regularizacao_cifar10.png",
                     "Efeito da regularização na MinhaCNN (CIFAR-10)")
    fig_train_vs_val(runs, "cifar10", ["MinhaCNN", "ResNet18 (fine-tuning)"],
                     "treino_validacao_cifar10.png",
                     "Treino x validação (acurácia) — CIFAR-10", split=True)
    fig_train_vs_val(runs, "mnist", ["MinhaCNN", "ResNet18 (fine-tuning)"],
                     "treino_validacao_mnist.png",
                     "Treino x validação (acurácia) — MNIST", split=True)
    main_labs = ["MinhaCNN", "ResNet18 (fine-tuning)"]
    for ds in ["mnist", "cifar10"]:
        fig_confusion(runs, ds, [l for l in main_labs if (l, ds) in runs])
        fig_per_class(runs, ds, [l for l in LABELS.values()
                                 if (l, ds) in runs and l in main_labs + ["ResNet18 (extração)"]])
        fig_errors(runs, ds)
    fig_cost(runs, "cifar10")
    print("figuras salvas em", FIG)


if __name__ == "__main__":
    main()
