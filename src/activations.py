"""DESAFIO: visualizar o que os kernels/neurônios estão aprendendo.

Gera em figures/:
  desafio_kernels_minhacnn.png        pesos da 1ª conv da MinhaCNN (MNIST e CIFAR-10)
  desafio_kernels_resnet18.png        conv1 da ResNet18: ImageNet x após fine-tuning
  desafio_mapas_ativacao_<ds>.png     mapas de ativação de cada bloco para uma imagem
  desafio_maximizacao_cifar10.png     imagens sintetizadas que maximizam canais de cada bloco
  desafio_classes_<ds>.png            imagens sintetizadas que maximizam cada classe
  desafio_top_imagens_cifar10.png     imagens do teste que mais ativam canais do bloco 3
  desafio_gradcam_cifar10.png         Grad-CAM: onde MinhaCNN e ResNet18 "olham"

Uso:  python -m src.activations
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from . import style
from .data import CLASSES, ROOT, get_splits
from .models import build_model

FIG = ROOT / "figures"
DSN = {"mnist": "MNIST", "cifar10": "CIFAR-10"}
DEV = "cuda"
torch.manual_seed(0); np.random.seed(0)


def load(tag, model, dataset):
    m = build_model(model, dataset, pretrained=False)
    sd = torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location=DEV, weights_only=True)
    m.load_state_dict(sd)
    return m.to(DEV).eval()


def to_img(t):
    """Tensor C,H,W em [0,1] -> array H,W(,3) para imshow."""
    a = t.detach().cpu().float().clamp(0, 1).numpy()
    return a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0)


def show(ax, t, title=None, cmap=None, **kw):
    a = to_img(t) if isinstance(t, torch.Tensor) else t
    ax.imshow(a, cmap=cmap or ("gray" if a.ndim == 2 else None), interpolation="nearest", **kw)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, color=style.INK2, fontweight="normal", pad=3)


def tile(ws: np.ndarray, rgb: bool, pad: int = 1):
    """Monta uma grade de kernels (N,C,k,k). Cada kernel é normalizado individualmente."""
    N, C, k, _ = ws.shape
    g = int(math.ceil(math.sqrt(N)))
    side = g * (k + pad) - pad
    canvas = np.full((side, side, 3) if rgb else (side, side), np.nan if not rgb else 1.0)
    if rgb:
        canvas[:] = np.array([0.988, 0.988, 0.984])
    for i, w in enumerate(ws):
        r, c = divmod(i, g)
        y, x = r * (k + pad), c * (k + pad)
        if rgb:
            t = w.transpose(1, 2, 0)
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        else:
            t = w[0] / (np.abs(w[0]).max() + 1e-8)
        canvas[y:y + k, x:x + k] = t
    return canvas


# --------------------------------------------------------------------------- #
# 1) Kernels da primeira camada
# --------------------------------------------------------------------------- #
def fig_kernels(m_mnist, m_cifar, r_ft):
    div = style.DIV.copy(); div.set_bad(style.SURFACE)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    wm = m_mnist.net.block1[0][0].weight.detach().cpu().numpy()
    wc = m_cifar.net.block1[0][0].weight.detach().cpu().numpy()
    show(axes[0], tile(wm, False), cmap=div, vmin=-1, vmax=1)
    axes[0].set_title("MNIST: 64 kernels 3×3 (1 canal)\nazul = peso negativo, vermelho = positivo",
                      fontsize=9)
    show(axes[1], tile(wc, True))
    axes[1].set_title("CIFAR-10: 64 kernels 3×3×3 (RGB)\ncada kernel normalizado para [0,1]",
                      fontsize=9)
    fig.suptitle("MinhaCNN: pesos da primeira camada convolucional", fontsize=12,
                 fontweight="bold", color=style.INK)
    fig.savefig(FIG / "desafio_kernels_minhacnn.png"); plt.close(fig)

    w_pre = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).conv1.weight.detach().numpy()
    w_ft = r_ft.net.conv1.weight.detach().cpu().numpy()
    rel = np.linalg.norm(w_ft - w_pre) / np.linalg.norm(w_pre)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    show(axes[0], tile(w_pre, True)); axes[0].set_title("Pesos do ImageNet (antes)", fontsize=9)
    show(axes[1], tile(w_ft, True)); axes[1].set_title("Após fine-tuning no CIFAR-10", fontsize=9)
    fig.suptitle(f"ResNet18: 64 kernels 7×7×3 da conv1  (variação relativa ‖ΔW‖/‖W‖ = {100 * rel:.1f}%)".replace(".", ","),
                 fontsize=12, fontweight="bold", color=style.INK)
    fig.savefig(FIG / "desafio_kernels_resnet18.png"); plt.close(fig)
    return float(rel)


# --------------------------------------------------------------------------- #
# 2) Mapas de ativação (feature maps) bloco a bloco
# --------------------------------------------------------------------------- #
def block_acts(model, x):
    acts, hooks = {}, []
    for i in (1, 2, 3):
        layer = getattr(model.net, f"block{i}")[1]
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out, k=i: acts.__setitem__(k, out.detach())))
    with torch.no_grad():
        logits = model(x)
    for h in hooks:
        h.remove()
    return acts, logits


def fig_feature_maps(model, x, label, dataset, n_ch=8):
    acts, logits = block_acts(model, x[None])
    pred = int(logits.argmax(1))
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(3, n_ch + 2, width_ratios=[1.8, 0.75] + [1] * n_ch, wspace=0.08,
                          hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    show(ax, x, cmap="gray")
    ax.set_title(f"Entrada\nclasse real: {CLASSES[dataset][label]}\nprevisto: "
                 f"{CLASSES[dataset][pred]}", fontsize=9, color=style.INK2, fontweight="normal")
    for r, k in enumerate((1, 2, 3)):
        a = acts[k][0]                                   # C,h,w
        top = a.mean((1, 2)).argsort(descending=True)[:n_ch]
        for c, ch in enumerate(top.tolist()):
            axc = fig.add_subplot(gs[r, c + 2])
            m = a[ch].cpu().numpy()
            show(axc, m, cmap=style.SEQ, vmin=0, vmax=max(m.max(), 1e-6))
            axc.set_title(f"canal {ch}", fontsize=7, color=style.MUTED, pad=2)
        lab = fig.add_subplot(gs[r, 1]); lab.axis("off")
        lab.text(0.95, 0.5, f"Bloco {k}\n{a.shape[1]}×{a.shape[2]}", ha="right", va="center",
                 fontsize=9, color=style.INK2, transform=lab.transAxes)
    fig.suptitle(f"MinhaCNN ({DSN[dataset]}): os 8 canais mais ativados em cada bloco "
                 "(azul escuro = ativação alta)", fontsize=11, fontweight="bold", color=style.INK)
    fig.savefig(FIG / f"desafio_mapas_ativacao_{dataset}.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
# 3) Maximização de ativação (imagens sintetizadas por gradiente ascendente)
# --------------------------------------------------------------------------- #
def blur(x):
    k = torch.tensor([1., 2., 1.], device=x.device)
    k = (k[:, None] * k[None, :]); k = (k / k.sum()).expand(x.shape[1], 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, groups=x.shape[1])


def act_max(model, target_fn, K, in_ch, size, steps=300, lr=0.08, tv=0.05, l2=0.02,
            use_blur=True, init=0.0, low_res=1, l1=0.0, jitter=2):
    """Otimiza K imagens simultaneamente; target_fn(img) devolve o objetivo (K,).

    Regularizadores (para gerar padrões legíveis em vez de ruído adversarial):
    variação total (tv), decaimento L2 para cinza, esparsidade L1 (fundo escuro no MNIST),
    desfoque gaussiano, deslocamentos aleatórios (jitter) e parametrização em baixa resolução.
    """
    r = size // low_res
    v = (init + 0.3 * torch.randn(K, in_ch, r, r, device=DEV)).requires_grad_()
    opt = torch.optim.Adam([v], lr=lr)

    def render(v):
        img = torch.sigmoid(v)
        if low_res > 1:
            img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
        return blur(img) if use_blur else img

    for s in range(steps):
        img = render(v)
        dx, dy = np.random.randint(-jitter, jitter + 1, 2)
        obj = target_fn(torch.roll(img, shifts=(int(dx), int(dy)), dims=(2, 3)))
        tv_loss = (img[..., 1:, :] - img[..., :-1, :]).abs().mean((1, 2, 3)) +                   (img[..., :, 1:] - img[..., :, :-1]).abs().mean((1, 2, 3))
        loss = (-obj + tv * tv_loss * obj.detach().abs().clamp(min=1)
                + l2 * ((img - 0.5) ** 2).mean((1, 2, 3))
                + l1 * img.mean((1, 2, 3)) * obj.detach().abs().clamp(min=1)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return render(v).detach()


def pick_channels(model, x_batch, block, n):
    acts, _ = block_acts(model, x_batch)
    a = acts[block].amax((2, 3))                        # B,C (máximo espacial)
    return a.std(0).argsort(descending=True)[:n].tolist()  # canais mais "seletivos"


def fig_act_max(model, x_batch, dataset, n=8):
    in_ch, size = x_batch.shape[1], x_batch.shape[-1]
    fig, axes = plt.subplots(3, n, figsize=(1.35 * n, 4.9))
    for r, k in enumerate((1, 2, 3)):
        chans = pick_channels(model, x_batch, k, n)
        # objetivo = ativação ANTES da ReLU (saída da BN), para que canais inativos no ponto de
        # partida ainda recebam gradiente; o clone evita a ReLU in-place que vem depois.
        bn = getattr(model.net, f"block{k}")[1][1]
        store = {}
        h = bn.register_forward_hook(lambda m, i, o: store.__setitem__("a", o.clone()))
        ch_t = torch.tensor(chans, device=DEV)

        def target(img):
            model(img)
            a = store["a"]
            return a[torch.arange(len(chans), device=DEV), ch_t].mean((1, 2))

        if k == 1:  # poucos passos: o campo receptivo é 5x5 e o padrão satura rápido
            imgs = act_max(model, target, n, in_ch, size, steps=60, lr=0.05, use_blur=False, tv=0.02)
        else:
            imgs = act_max(model, target, n, in_ch, size, use_blur=True, tv=0.05)
        h.remove()
        for c in range(n):
            show(axes[r, c], imgs[c], cmap="gray")
            axes[r, c].set_title(f"canal {chans[c]}", fontsize=7, color=style.MUTED, pad=2)
        axes[r, 0].set_ylabel(f"Bloco {k}", fontsize=9, color=style.INK2, rotation=0,
                              ha="right", va="center", labelpad=10)
    fig.suptitle(f"MinhaCNN ({DSN[dataset]}): padrão de entrada que mais excita cada canal\n"
                 "(maximização de ativação por gradiente ascendente na imagem)",
                 fontsize=11, fontweight="bold", color=style.INK)
    fig.tight_layout()
    fig.savefig(FIG / f"desafio_maximizacao_{dataset}.png"); plt.close(fig)


def fig_class_max(model, dataset, in_ch, size):
    cls = torch.arange(10, device=DEV)

    def target(img):
        return model(img)[torch.arange(10, device=DEV), cls]

    if dataset == "mnist":  # traço claro sobre fundo escuro, como nos dígitos reais
        imgs = act_max(model, target, 10, in_ch, size, steps=500, lr=0.05, tv=0.15, l2=0.0,
                       init=-2.0, low_res=2, l1=0.6, jitter=1)
    else:
        imgs = act_max(model, target, 10, in_ch, size, steps=500, lr=0.05, tv=0.12, l2=0.05,
                       low_res=2, jitter=2)
    with torch.no_grad():
        conf = model(imgs).softmax(1)[torch.arange(10), cls].cpu().numpy()
    fig, axes = plt.subplots(1, 10, figsize=(13, 1.9))
    for c in range(10):
        show(axes[c], imgs[c], cmap="gray")
        axes[c].set_title(f"{CLASSES[dataset][c]}\np={conf[c]:.2f}", fontsize=8,
                          color=style.INK2, pad=3)
    fig.suptitle(f"MinhaCNN ({DSN[dataset]}): imagem sintetizada que maximiza cada classe",
                 fontsize=11, fontweight="bold", color=style.INK, y=1.04)
    fig.savefig(FIG / f"desafio_classes_{dataset}.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
# 4) Imagens reais que mais ativam canais profundos
# --------------------------------------------------------------------------- #
def fig_top_images(model, x_test, y_test, dataset="cifar10", n_ch=6, n_img=8):
    maxs = []
    for k in range(0, len(x_test), 500):
        xb = x_test[k:k + 500].float().div(255).to(DEV)
        acts, _ = block_acts(model, xb)
        maxs.append(acts[3].amax((2, 3)).cpu())
    maxs = torch.cat(maxs)                             # N,C
    top = maxs.argsort(0, descending=True)[:50]       # 50 imagens que mais ativam cada canal
    y = y_test.numpy()
    purity = torch.tensor([np.bincount(y[top[:, c].numpy()], minlength=10).max() / 50
                           for c in range(maxs.shape[1])])
    chans = purity.argsort(descending=True)[:n_ch].tolist()
    fig, axes = plt.subplots(n_ch, n_img, figsize=(1.25 * n_img, 1.35 * n_ch))
    for r, ch in enumerate(chans):
        idx = top[:n_img, ch].numpy()
        dom = np.bincount(y[top[:, ch].numpy()], minlength=10).argmax()
        for c, i in enumerate(idx):
            show(axes[r, c], x_test[i].float().div(255))
        axes[r, 0].set_ylabel(f"canal {ch}\n{purity[ch]:.0%} {CLASSES[dataset][dom]}",
                              fontsize=8, color=style.INK2, rotation=0, ha="right",
                              va="center", labelpad=8)
    fig.suptitle("MinhaCNN (CIFAR-10), bloco 3: imagens do teste que mais ativam cada canal\n"
                 "(canais mais seletivos; % = fração da classe dominante entre as 50 imagens de topo)",
                 fontsize=10, fontweight="bold", color=style.INK)
    fig.tight_layout()
    fig.savefig(FIG / f"desafio_top_imagens_{dataset}.png"); plt.close(fig)
    return {int(c): float(purity[c]) for c in chans}, float(purity.mean())


# --------------------------------------------------------------------------- #
# 5) Grad-CAM
# --------------------------------------------------------------------------- #
def gradcam(model, layer, x):
    store = {}

    def fh(m, i, o):
        o.retain_grad(); store["a"] = o

    h = layer.register_forward_hook(fh)
    model.zero_grad(set_to_none=True)
    out = model(x)
    pred = out.argmax(1)
    out[torch.arange(len(x)), pred].sum().backward()
    h.remove()
    a, g = store["a"], store["a"].grad
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam / (cam.amax((2, 3), keepdim=True) + 1e-8)
    return cam.detach(), pred.detach()


def fig_gradcam(m_cifar, r_ft, x_test, y_test, n=8):
    y = y_test.numpy()
    idx = [int(np.where(y == c)[0][1]) for c in (3, 7, 8, 9, 2, 6, 0, 5)][:n]
    x = x_test[idx].float().div(255).to(DEV)
    cam_m, p_m = gradcam(m_cifar, m_cifar.net.block3[1], x)
    cam_r, p_r = gradcam(r_ft, r_ft.net.layer4, x)
    up = lambda t: F.interpolate(t, size=(128, 128), mode="bicubic", align_corners=False).clamp(0, 1)
    xs, cm, cr = up(x), up(cam_m), up(cam_r)
    fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 5.6))
    cls = CLASSES["cifar10"]
    for c in range(n):
        show(axes[0, c], xs[c], title=f"real: {cls[y[idx[c]]]}")
        show(axes[1, c], xs[c] * (0.12 + 0.88 * cm[c]), title=f"prev.: {cls[p_m[c]]}")
        show(axes[2, c], xs[c] * (0.12 + 0.88 * cr[c]), title=f"prev.: {cls[p_r[c]]}")
    for r, name in enumerate(["Imagem", "MinhaCNN\n(bloco 3, 8×8)", "ResNet18\n(layer4, 7×7)"]):
        axes[r, 0].set_ylabel(name, fontsize=9, color=style.INK2, rotation=0, ha="right",
                              va="center", labelpad=10)
    fig.suptitle("Grad-CAM no CIFAR-10: regiões claras são as que mais pesaram na decisão",
                 fontsize=11, fontweight="bold", color=style.INK)
    fig.tight_layout()
    fig.savefig(FIG / "desafio_gradcam_cifar10.png"); plt.close(fig)


def main():
    style.setup()
    FIG.mkdir(exist_ok=True)
    m_mnist = load("minhacnn_mnist", "minhacnn", "mnist")
    m_cifar = load("minhacnn_cifar10", "minhacnn", "cifar10")
    r_ft = load("resnet18_ft_cifar10", "resnet18", "cifar10")
    s_m, s_c = get_splits("mnist"), get_splits("cifar10")
    xm, ym = s_m["test"]; xc, yc = s_c["test"]

    rel = fig_kernels(m_mnist, m_cifar, r_ft)
    print("kernels ok; variação relativa conv1 ResNet:", rel)
    fig_feature_maps(m_cifar, xc[int(np.where(yc.numpy() == 7)[0][0])].float().div(255).to(DEV),
                     7, "cifar10")
    fig_feature_maps(m_mnist, xm[0].float().div(255).to(DEV), int(ym[0]), "mnist")
    print("mapas de ativação ok")
    fig_act_max(m_cifar, xc[:1000].float().div(255).to(DEV), "cifar10")
    fig_act_max(m_mnist, xm[:1000].float().div(255).to(DEV), "mnist")
    fig_class_max(m_mnist, "mnist", 1, 28)
    fig_class_max(m_cifar, "cifar10", 3, 32)
    print("maximização ok")
    sel, mean_purity = fig_top_images(m_cifar, xc, yc)
    print("top imagens ok", sel, "pureza média de todos os canais:", round(mean_purity, 3))
    fig_gradcam(m_cifar, r_ft, xc, yc)
    print("grad-cam ok")
    import json
    (ROOT / "results" / "desafio.json").write_text(json.dumps(
        {"resnet_conv1_rel_change": rel, "top_channels_purity": sel,
         "mean_channel_purity_block3": mean_purity}, indent=1))


if __name__ == "__main__":
    main()
