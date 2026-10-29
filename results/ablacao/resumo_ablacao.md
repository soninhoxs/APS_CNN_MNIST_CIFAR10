| Dataset | Configuração | Sementes | Acurácia teste (%) | Erros (de 10.000) | F1 macro | Acc. treino (%) | Gap (p.p.) | Perda teste | Melhor época |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | Com aumento de dados | 3 | 99,68 ± 0,02 | 32 ± 2 | 0,9968 ± 0,0002 | 99,84 ± 0,03 | 0,15 ± 0,03 | 0,097 ± 0,000 | 13 ± 1 |
| MNIST | Sem aumento de dados | 3 | 99,70 ± 0,02 | 30 ± 2 | 0,9970 ± 0,0002 | 99,97 ± 0,01 | 0,26 ± 0,01 | 0,097 ± 0,001 | 14 ± 1 |
| CIFAR-10 | Com aumento de dados | 3 | 92,22 ± 0,18 | 778 ± 18 | 0,9219 ± 0,0018 | 98,19 ± 0,04 | 5,96 ± 0,21 | 0,303 ± 0,001 | 40 ± 1 |
| CIFAR-10 | Sem aumento de dados | 3 | 89,97 ± 0,20 | 1003 ± 20 | 0,8992 ± 0,0021 | 100,00 ± 0,00 | 10,03 ± 0,20 | 0,389 ± 0,004 | 38 ± 1 |
| CIFAR-10 | Sem nenhuma regularização | 1 | 86,17 | 1383 | 0,8616 | 100,00 | 13,83 | 0,765 | 27 |
| CIFAR-10 | Só espelhamento | 1 | 91,86 | 814 | 0,9183 | 99,88 | 8,02 | 0,322 | 40 |
| CIFAR-10 | Só recorte (±4 px) | 1 | 91,33 | 867 | 0,9127 | 99,79 | 8,46 | 0,335 | 39 |
| CIFAR-10 | Recorte ±2 px + espelhamento | 1 | 92,31 | 769 | 0,9227 | 98,68 | 6,37 | 0,302 | 38 |
| CIFAR-10 | Recorte ±6 px + espelhamento | 1 | 91,99 | 801 | 0,9196 | 97,40 | 5,41 | 0,309 | 39 |
| CIFAR-10 | Recorte ±8 px + espelhamento | 1 | 91,58 | 842 | 0,9154 | 96,87 | 5,29 | 0,327 | 39 |
