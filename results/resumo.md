| Dataset | Modelo | Parâmetros | Épocas | Acurácia teste | F1 macro | Acc. treino | Gap | Tempo treino | Inferência |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | MinhaCNN | 1.18 M | 15 | 99.66% | 0.9966 | 99.82% | 0.16 pp | 2.1 min | 25,064 img/s |
| MNIST | ResNet18 (fine-tuning) | 11.18 M | 3 | 99.53% | 0.9953 | 99.71% | 0.18 pp | 5.8 min | 1,573 img/s |
| MNIST | ResNet18 (extração) | 11.18 M | 5 | 96.31% | 0.9627 | 96.59% | 0.28 pp | 3.8 min | 1,571 img/s |
| CIFAR-10 | MinhaCNN | 1.18 M | 40 | 92.43% | 0.9240 | 98.16% | 5.73 pp | 6.1 min | 19,235 img/s |
| CIFAR-10 | ResNet18 (fine-tuning) | 11.18 M | 5 | 95.46% | 0.9545 | 98.68% | 3.22 pp | 8.0 min | 1,567 img/s |
| CIFAR-10 | ResNet18 (extração) | 11.18 M | 5 | 86.97% | 0.8691 | 88.55% | 1.58 pp | 2.8 min | 1,568 img/s |
| CIFAR-10 | ResNet18 do zero | 11.18 M | 5 | 86.92% | 0.8686 | 89.62% | 2.70 pp | 7.9 min | 1,568 img/s |
| CIFAR-10 | MinhaCNN sem regularização | 1.18 M | 40 | 86.17% | 0.8616 | 100.00% | 13.83 pp | 6.0 min | 19,043 img/s |
