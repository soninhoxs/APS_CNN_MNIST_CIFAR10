"""Baixa os conjuntos de dados para data/raw/.

    python download_data.py

- MNIST: Parquet oficial do Hugging Face (https://huggingface.co/datasets/ylecun/mnist)
- CIFAR-10: espelho oficial no Hugging Face (https://huggingface.co/datasets/uoft-cs/cifar10),
  mesmo conteúdo e ordem da versão de https://www.cs.toronto.edu/~kriz/cifar.html.
  Se preferir a versão de Toronto, baixe cifar-10-binary.tar.gz e extraia em
  data/raw/cifar10/ (o código usa automaticamente a versão binária quando ela existe).
"""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data" / "raw"
FILES = {
    "mnist/train-00000-of-00001.parquet":
        "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet",
    "mnist/test-00000-of-00001.parquet":
        "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet",
    "cifar10_hf/train-00000-of-00001.parquet":
        "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
    "cifar10_hf/test-00000-of-00001.parquet":
        "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
}

if __name__ == "__main__":
    for rel, url in FILES.items():
        dst = ROOT / rel
        if dst.exists():
            print("já existe:", dst); continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("baixando", url)
        urllib.request.urlretrieve(url, dst)
    print("ok")
