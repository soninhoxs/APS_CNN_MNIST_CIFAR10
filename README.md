# APS — Redes Convolucionais: CNN própria x ResNet18 pré-treinada (MNIST e CIFAR-10)

Atividade Prática Supervisionada da disciplina **Introdução à Aprendizagem Profunda**.
Implementamos uma CNN própria em PyTorch (**MinhaCNN**, ~1,2 M de parâmetros, com regularização),
treinada do zero no MNIST e no CIFAR-10, e comparamos com a **ResNet18 pré-treinada no ImageNet**
(fine-tuning completo, extração de características e um controle treinado do zero).
Também resolvemos o **desafio**: visualização de kernels, mapas de ativação, maximização de ativação,
imagens que mais ativam cada neurônio e Grad-CAM.

📄 **Relatório:** [`relatorio/Relatorio_APS_CNN.pdf`](relatorio/Relatorio_APS_CNN.pdf)
(também em [DOCX](relatorio/Relatorio_APS_CNN.docx) e [Markdown](relatorio/relatorio.md)).

## Resultados (conjunto de teste)

| Dataset | Modelo | Parâmetros (treináveis) | Épocas | Acurácia teste | F1 macro | Acc. treino | Gap | Tempo de treino | Inferência (img/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | MinhaCNN | 1,18 M | 15 | **99,66%** | 0,9966 | 99,82% | 0,2 p.p. | 2,1 min | 25.064 |
| MNIST | ResNet18 (fine-tuning) | 11,18 M | 3 | **99,53%** | 0,9953 | 99,71% | 0,2 p.p. | 5,8 min | 1.573 |
| MNIST | ResNet18 (extração) | 11,18 M (5.130) | 5 | 96,31% | 0,9627 | 96,59% | 0,3 p.p. | 3,8 min | 1.571 |
| CIFAR-10 | MinhaCNN | 1,18 M | 40 | **92,43%** | 0,9240 | 98,16% | 5,7 p.p. | 6,1 min | 19.235 |
| CIFAR-10 | ResNet18 (fine-tuning) | 11,18 M | 5 | **95,46%** | 0,9545 | 98,68% | 3,2 p.p. | 8,0 min | 1.567 |
| CIFAR-10 | ResNet18 (extração) | 11,18 M (5.130) | 5 | 86,97% | 0,8691 | 88,55% | 1,6 p.p. | 2,8 min | 1.568 |
| CIFAR-10 | ResNet18 do zero | 11,18 M | 5 | 86,92% | 0,8686 | 89,62% | 2,7 p.p. | 7,9 min | 1.568 |
| CIFAR-10 | MinhaCNN sem regularização | 1,18 M | 40 | 86,17% | 0,8616 | 100,00% | 13,8 p.p. | 6,0 min | 19.043 |

- **MNIST:** MinhaCNN 99,66% x ResNet18 (fine-tuning) 99,53% — MinhaCNN ligeiramente melhor (p = 0,041, McNemar).
- **CIFAR-10:** ResNet18 (fine-tuning) 95,46% x MinhaCNN 92,43% (p < 0,001). A mesma ResNet18 **sem** pré-treino fica em 86,92%: a vantagem vem do ImageNet.
- A MinhaCNN é 9,5× menor e 12× mais rápida na inferência; sem regularização ela cai para 86,17% no CIFAR-10.

![Erro de teste](figures/comparacao_erro_teste.png)
![Neurônios do bloco 3](figures/desafio_top_imagens_cifar10.png)


## Estrutura

```
├── run_all.py            # roda todos os experimentos + figuras (ponto de entrada)
├── download_data.py      # baixa MNIST e CIFAR-10 para data/raw/
├── src/
│   ├── data.py           # leitura dos datasets, divisão treino/val/teste, aumento de dados na GPU
│   ├── models.py         # MinhaCNN e ResNet18 (torchvision, pesos ImageNet)
│   ├── train.py          # treino/avaliação de um experimento (CLI)
│   ├── plots.py          # gráficos e tabelas de comparação
│   ├── activations.py    # DESAFIO: kernels, ativações, maximização, top imagens, Grad-CAM
│   ├── style.py          # paleta e estilo dos gráficos
│   └── make_report.py    # relatorio.md -> DOCX e PDF
├── results/<experimento>/ # history.json (curvas), metrics.json (métricas), test_preds.npz
├── results/resumo.csv    # tabela-resumo de todos os experimentos
├── figures/              # todos os gráficos (PNG)
├── logs/                 # saída de cada treino
└── relatorio/            # relatório (MD, DOCX, PDF, HTML)
```

## Como reproduzir

Requer GPU NVIDIA com CUDA (testado numa RTX 3050 Laptop de 6 GB; tempo total ≈ 1 h).

```bash
pip install -r requirements.txt     # ou use um ambiente conda com PyTorch + CUDA
python download_data.py             # MNIST e CIFAR-10 (Hugging Face)
python run_all.py                   # treina tudo, gera figuras e visualizações
python -m src.make_report           # (opcional) regenera DOCX/PDF a partir do relatorio.md
```

Rodar um experimento isolado:

```bash
python -m src.train --dataset cifar10 --model minhacnn --epochs 40 --lr 0.1 --tag minhacnn_cifar10
python -m src.train --dataset cifar10 --model resnet18 --epochs 5 --lr 0.01 --wd 1e-4 --tag resnet18_ft_cifar10
python -m src.train --dataset cifar10 --model resnet18 --freeze --epochs 5 --lr 0.1 --wd 1e-4 --tag resnet18_frozen_cifar10
python -m src.train --dataset cifar10 --model minhacnn --epochs 40 --lr 0.1 --no-reg --tag minhacnn_noreg_cifar10
```

Os pesos treinados ficam em `checkpoints/` (fora do git por tamanho).

> **Windows + Anaconda:** se aparecer `OMP: Error #15`, é o conflito conhecido entre as cópias da
> runtime OpenMP do numpy (MKL) e do PyTorch. O pacote `src` já define `KMP_DUPLICATE_LIB_OK=TRUE`.

## Fontes dos dados

- MNIST: <https://huggingface.co/datasets/ylecun/mnist>
- CIFAR-10: <https://www.cs.toronto.edu/~kriz/cifar.html> (usamos o espelho oficial
  <https://huggingface.co/datasets/uoft-cs/cifar10>; se `data/raw/cifar10/cifar-10-batches-bin/`
  existir, o código usa a versão binária de Toronto)
- ResNet18: <https://docs.pytorch.org/vision/main/models.html> (`ResNet18_Weights.IMAGENET1K_V1`)
