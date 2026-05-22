# Visão Computacional — Classificação de Placas de Trânsito (GTSRB)

Trabalho da pós-graduação em Visão Computacional: classificação das **43 classes**
de placas de trânsito do dataset **GTSRB** (German Traffic Sign Recognition Benchmark)
com CNNs treinadas **do zero** (sem pesos pré-treinados).

O projeto parte de uma CNN baseline e avança por uma série de experimentos
controlados (otimizadores, batch normalization, data augmentation, combinações e
arquiteturas/regularização opcionais), sempre com split fixo (`seed=42`,
80% treino / 20% validação) e o test set reservado apenas para a avaliação final.

## Estrutura

```
.
├── baseline.ipynb            # CNN simples (2 conv + 2 FC), SGD
├── exp1_otimizadores.ipynb   # SGD vs SGD+momentum vs Adam
├── exp2_batchnorm.ipynb      # sem BN vs BN-conv vs BN-conv+fc (× SGD-mom/Adam)
├── exp3_augmentation.ipynb   # sem aug vs light vs medium vs strong
├── exp4_combinacoes.ipynb    # ablação combinando img64, weighted loss e aug
├── exp5_opcionais.ipynb      # VGG-like, ResNet pequena, dropout/weight-decay/early-stopping
├── src/gtsrb.py              # Carregamento dos dados + exportação de predições (não modificar)
├── exemplo.py                # Uso mínimo do kit
├── checkpoints/              # Pesos .pt e summaries .json de cada experimento
├── results/                  # CSVs de predição no test set (um por experimento)
├── figs/                     # Figuras geradas pelos notebooks
├── Dockerfile                # PyTorch 2.6 + CUDA 12.4, Jupyter Lab
├── docker-compose.yml        # Serviço gtsrb com GPU
└── requirements.txt
```

> O dataset (`data/`) **não** está versionado por causa do tamanho (~690 MB).
> Baixe o GTSRB e coloque em `data/gtsrb/` antes de rodar os notebooks.

## Experimentos e melhores resultados

Acurácia de **validação** do melhor checkpoint de cada rodada (acurácia de teste
final em `results/*.csv`):

| Experimento | Melhor configuração | Melhor val. acc |
|---|---|---|
| Baseline | CNN simples + SGD | — (ver `baseline.ipynb`) |
| Exp 1 — Otimizadores | Adam | 0,9944 |
| Exp 2 — Batch Norm | BN-conv+fc + SGD-momentum | 0,9974 |
| Exp 3 — Augmentation | aug medium/strong | 0,9992 |
| Exp 4 — Combinações | img64 + aug | 0,9991 |
| Exp 5 — Opcionais | VGG-like pequena | 0,9994 |

## Como rodar (Docker)

```bash
# Sobe o Jupyter Lab com GPU em http://localhost:8888
JUPYTER_TOKEN=seu_token docker compose up --build

# Rodar um script ad-hoc dentro do container
docker compose run --rm gtsrb python exemplo.py
```

O serviço monta o diretório atual em `/workspace` e usa todas as GPUs NVIDIA
disponíveis (driver >= 550 para CUDA 12.4).

## Como rodar (local)

```bash
pip install -r requirements.txt
jupyter lab
```

## Uso do kit de dados

```python
from src.gtsrb import get_dataloaders, save_predictions, NUM_CLASSES, GTSRB_CLASSES

# split fixo: seed=42, 80% treino / 20% validação
train_loader, val_loader, test_loader = get_dataloaders(img_size=32, batch_size=128)

# data augmentation: passe seu próprio transform de treino
from torchvision import transforms
meu_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                         std=[0.2724, 0.2608, 0.2669]),
])
train_loader, val_loader, test_loader = get_dataloaders(
    img_size=32, batch_size=128, train_transform=meu_transform,
)
```

## Predições / entrega

Cada experimento gera um CSV de predições no test set (12.630 linhas):

```python
save_predictions(y_pred, "results/predicoes_baseline.csv", experiment_name="Baseline CNN SGD")
```

```
image_index,predicted_class
0,14
1,1
2,38
...
12629,35
```

## Regras do trabalho

1. **Não modifique** `src/gtsrb.py` — garante o mesmo split e formato para todos.
2. O test set só deve ser usado para a **avaliação final**.
3. Todos os modelos são **treinados do zero** (sem pesos pré-treinados).
