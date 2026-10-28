import os

# No Windows + Anaconda, o numpy (MKL) e o PyTorch trazem cada um sua cópia da
# runtime OpenMP (libiomp5md.dll); sem isto o matplotlib aborta com "OMP: Error #15".
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
