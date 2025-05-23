
# Seismic Facies Segmentation Framework

## Overview

This repository provides a systematic framework for evaluating state-of-the-art deep learning models on **Seismic Facies Classification (SFC)** tasks. The implementation is based on the methodology proposed in the paper:

> Gabriel Lima, Gabriel Amarante, Willian Barreiros Jr., Matheus T. P. Souza, Wagner Meira Jr., Renato Ferreira, George Teodoro.  
> **"A Systematic Evaluation Methodology of Deep Learning on Seismic Facies Classification"**  
> Universidade Federal de Minas Gerais (UFMG), Brazil.

The framework supports multiple segmentation models (traditional and lightweight), a variety of seismic datasets, and different data partitioning strategies for a fair and reproducible evaluation.

## Key Features

- Support for 9 deep learning segmentation architectures:
  - UNet
  - UNet 3+
  - Attention UNet
  - Flexible BridgeNet
  - CFPNetM
  - ENet
  - ESPNet
  - ICNet
  - EfficientNetB1
- Systematic data partitioning strategies:
  - Large Rectangular Prisms (LRP)
  - Rectangular Prisms with Varying Sizes (RPRV)
  - Rectangular Prisms with Equally Distant Slices (RPEDS)
  - Equally Distant Slices (EDS)
- Evaluation on **Parihaka** and **Penobscot** public seismic datasets.
- Flexible model selection, hyperparameters, and GPU configuration.
- Support for **Focal Loss** and **Cross-Entropy Loss**.

---

## Environment Setup

1. Clone this repository:

```bash
git clone <repository-url>
cd Seismic-Segmentation-Framework
```

2. Create a conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate seismic_facies
```

## Running the Code

You can run the training and evaluation script using:

```bash
python train.py [OPTIONS]
```

For example:

```bash
python train.py --model 0 --dataset 0 --epochs 100 --batch_size 16 --gpuID 0
```

The results (graphs and performance tables) will be saved under:

```
./results/<folder>
```

Model checkpoints will be saved under:

```
./checkpoints/<folder>
```

---

## Command-Line Arguments

| Flag | Type | Default | Description |
|-------|------|---------|-------------|
| `--optimizer, -o` | int | 0 | Optimizer: `0` = Adam, `1` = SGD, `2` = RMSprop |
| `--loss, -l` | int | 0 | Loss: `0` = Cross Entropy, `1` = Focal Loss |
| `--gamma, -g` | float | 3.6 | Gamma parameter for Focal Loss |
| `--epochs, -e` | int | 100 | Number of training epochs |
| `--patience, -p` | int | 15 | Early stopping patience |
| `--delta, -d` | float | 1e-4 | Minimum delta for early stopping |
| `--batch_size, -b` | int | 16 | Training batch size |
| `--kernel` | int | 7 | Kernel size for convolutions |
| `--model, -m` | int | 0 | Model selection: `0`=UNet, `1`=UNet3+, `2`=Attention UNet, `3`=Flexible BridgeNet, `4`=CFPNetM, `5`=ENet, `6`=ESPNet, `7`=ICNet, `8`=EfficientNetB1 |
| `--filters` | int | 6 | Number of convolutional filters |
| `--folder, -f` | str | "default_folder" | Output folder for results |
| `--name, -n` | str | "default" | Model name for saving outputs |
| `--gpuID` | int | 1 | GPU ID (`-1` for CPU) |
| `--dataset` | int | 0 | Dataset selection: `0`=Parihaka LRP, `1`=Penobscot LRP, `2`=Parihaka RPRV, ..., up to `7`=Penobscot EDS |
| `--slice_height, -s1` | int | 992 | Height of image slices |
| `--slice_width, -s2` | int | 192 | Width of image slices |
| `--train_val_stride_height` | int | 128 | Stride in height for train/val patches |
| `--train_val_stride_width` | int | 64 | Stride in width for train/val patches |
| `--test_stride_height` | int | 128 | Stride in height for test patches |
| `--test_stride_width` | int | 64 | Stride in width for test patches |
| `--train_limit_x` | int | 192 | X dimension limit for RPRV/RPEDS |
| `--train_limit_y` | int | 192 | Y dimension limit for RPRV/RPEDS |
| `--extra_train_slices` | int | 2 | Number of extra slices for RPEDS/EDS |

---

## Data Partitioning Strategies

### 1. **LRP** - Large Rectangular Prisms
- Classical approach with contiguous training, validation, and test blocks.
- High annotation effort but strong performance.

### 2. **RPRV** - Rectangular Prisms with Varying Sizes
- Reduces training data size compared to LRP.
- Simulates real-world low-data scenarios.

### 3. **RPEDS** - Rectangular Prisms with Equally Distant Slices
- Adds global context using sparse slices.
- Reduces annotation needs while improving accuracy.

### 4. **EDS** - Equally Distant Slices
- Purely slice-based training across the entire volume.
- Achieves competitive results with 13x less training data compared to LRP.

---

## Model Selection

| Model | Type | FLOPs | Params |
|--------|------|-------|--------|
| UNet | Heavy | High | 116M |
| BridgeNet | Heavy | High | 105M |
| UNet 3+ | Heavy | Very High | 76M |
| Attention UNet | Heavy | Very High | 275M |
| EfficientNet B1 | Light | Low | 9.9M |
| ESPNet | Light | Very Low | 0.57M |
| ENet | Light | Very Low | 0.36M |
| ICNet | Light | Low | 6.7M |
| CFPNet-M | Light | Very Low | 0.64M |

---

## Outputs

- **Graphs:** Loss and accuracy curves for training and validation.
- **Tables:** Summary of model configurations and performance metrics.
- **Checkpoints:** Saved model weights for best validation accuracy.
- **Predictions:** Model predictions on test data.

---

## Example Output Structure

```
├── checkpoints/
│   └── <folder>/
│       └── checkpoint_<name>.weights.h5
├── results/
│   └── <folder>/
│       ├── graphs/
│       │   └── graph_<name>.png
│       └── tables/
│           └── table_<name>.txt
```

---

## Evaluation Metrics

- **Accuracy:** Overall correct predictions.
- **Macro F1-Score:** Harmonic mean of precision and recall, addressing class imbalance.

---

## Recommended Datasets

1. **Parihaka**  
   - 6 facies  
   - Continental slope and submarine canyons.

2. **Penobscot**  
   - 8 facies  
   - Shallow-marine carbonate environments.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{lima2025systematic,
  title={A Systematic Evaluation Methodology of Deep Learning on Seismic Facies Classification},
  author={Lima, Gabriel and Amarante, Gabriel and Barreiros Jr., Willian and Souza, Matheus T. P. and Meira Jr., Wagner and Ferreira, Renato and Teodoro, George},
  year={2025},
  institution={UFMG, Brazil}
}
```

---

## Acknowledgements

This work is based on research conducted at the **Department of Computer Science, UFMG**, and leverages public seismic datasets made available by SEG.

---

## License

[MIT License](LICENSE)
