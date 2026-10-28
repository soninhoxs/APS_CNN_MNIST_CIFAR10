"""Estilo comum dos gráficos (paleta categórica validada p/ daltonismo + rampas)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Cor segue a entidade (o modelo), nunca a posição no gráfico.
COLORS = {
    # Ordem fixa da paleta (validada: CVD ΔE >= 9 entre vizinhas). Contraste < 3:1
    # nos slots 3-5 -> os gráficos levam rótulos diretos e o relatório traz tabelas.
    "MinhaCNN": "#2a78d6",                 # 1 azul
    "ResNet18 (fine-tuning)": "#eb6834",   # 2 laranja
    "ResNet18 (extração)": "#1baf7a",      # 3 verde-água
    "ResNet18 do zero": "#eda100",         # 4 amarelo
    "MinhaCNN sem regularização": "#e87ba4",  # 5 magenta
}

BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec",
                                                         "#ec8a89", "#b52c2c"])


def setup():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "lines.linewidth": 2,
    })
