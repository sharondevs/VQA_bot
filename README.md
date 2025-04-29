# Vision-Language Modeling for Real-World Accessibility

*A CLIP-Based Visual Question Answering (VQA) System for the VizWiz Dataset*

---

## Table of Contents
1. [Introduction](#introduction)
2. [Demo](#demo)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Training & Evaluation](#training--evaluation)
8. [Results](#results)
9. [Model Architecture](#model-architecture)
10. [Pre-Processing Pipeline](#pre-processing-pipeline)
11. [Reproducing the Paper Figures](#reproducing-the-paper-figures)
12. [Known Limitations & Future Work](#known-limitations--future-work)
13. [Citation](#citation)
14. [License](#license)
15. [Contributors](#contributors)

---

## Introduction

This repository contains the full code & assets for our course project **“Vision-Language Modeling for Real-World Accessibility: A CLIP-Based VQA System for VizWiz.”**  
We combine the powerful **CLIP ViT-B/32** vision–language encoder with a lightweight multi-head classifier to answer questions about images captured **by blind or low-vision users** in everyday settings.  
Our approach follows *“Less Is More: CLIP-Based Simple and Efficient VQA”*—freezing the CLIP backbone and learning only a small classifier on top.  
A detailed write-up of the method, experiments, and analysis is provided in [`project_report.pdf`](./project_report_sumanman_vpoliset_ssaseend.pdf).

<p align="center">
  <img src="docs/architecture.svg" width="650" alt="CLIP-VQA architecture" />
</p>

---

## Demo

| Notebook | Description |
|----------|-------------|
| [`notebooks/VizWiz_Demo.ipynb`](notebooks/VizWiz_Demo.ipynb) | End-to-end demo – load an image + question, run the model, visualise the answer & attention heat-map |

Run the notebook on **GPU** (e.g.  
Google Colab) for best performance.

---

## Dataset

This project uses the **VizWiz VQA** dataset:

* **≈ 31 k** image–question pairs collected from blind/low-vision photographers.
* Each question has **10 crowd-sourced answers** plus an *answerability* flag.
* Split: **train / val / test** (official JSON annotations).  

```bash
# Download original images (≈ 20 GB) & annotations
python scripts/download_vizwiz.py --output_dir data/vizwiz
```
> **Note:** The download script requires Kaggle credentials or direct links provided by the dataset maintainers.

---

## Project Structure

```
├── checkpoints/           # Model checkpoints (.pt)
├── configs/               # YAML config files for experiments
├── data/                  # (place VizWiz images & annotations here)
├── docs/                  # Figures used in the report / README
├── notebooks/             # Jupyter notebooks
├── src/
│   ├── clip_vizwiz.py     # CLIP feature extractor wrapper
│   ├── dataset.py         # VizWizDataset 🛈 for PyTorch
│   ├── model.py           # Multi-head classifier (answer, type, answerability)
│   ├── train.py           # Training loop
│   └── evaluate.py        # Evaluation + VizWiz metrics
└── requirements.txt       # Python dependencies
```

---

## Installation

> Tested on **Python 3.10** & **PyTorch 2.2** with CUDA 11.8.

```bash
# clone repo
$ git clone https://github.com/<your-username>/<repo>.git
$ cd <repo>

# create & activate virtual environment (optional)
$ python -m venv .venv
$ source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# install deps
$ pip install -r requirements.txt
```

Key libraries:
* `torch`, `torchvision`, `torchaudio`
* `openai-clip` (via `git+https://github.com/openai/CLIP`)  
* `tqdm`, `pandas`, `scikit-learn`, `matplotlib`

---

## Quick Start

**1. Prepare data** (see [Dataset](#dataset)).  
**2. Train:**
```bash
python src/train.py   --config configs/vqa_clip_vitb32.yaml   --data_root data/vizwiz   --output_dir checkpoints/vitb32_run1
```
**3. Evaluate:**
```bash
python src/evaluate.py   --checkpoint checkpoints/vitb32_run1/best.pt   --split val
```
Sample output:
```
VizWiz Accuracy : 61.5 %
Overall Accuracy: 48.0 %
Answerability   : 79.8 %
```

---

## Training & Evaluation

Training parameters can be changed via YAML in `configs/` (backbone, lr, epochs, batch_size, losses).  
We train for **90 epochs** with **Adam (lr = 5e-4)** on a single RTX A4000 16 GB.  The best checkpoint (epoch 85) is selected by **val VizWiz Accuracy**.

To reproduce paper-quality plots (loss & accuracy curves):
```bash
python scripts/plot_curves.py --log_dir checkpoints/vitb32_run1/logs
```

---

## Results

| Metric | Train | Val |
|--------|-------|-----|
| **VizWiz Accuracy** | **80.4 %** | **61.5 %** |
| Accuracy | 76.4 % | 48.0 % |
| Answerability | 80.2 % | 79.8 % |

The performance gap reflects real-world noise (blur, occlusion) in VizWiz.  Scaling to **CLIP ViT-L/14** and training on the full dataset is expected to further boost validation metrics.

---

## Model Architecture

> Full details in Section 4 of the [project report](./project_report_sumanman_vpoliset_ssaseend.pdf).

* **Feature extraction:** CLIP ViT-B/32 (frozen)
* **Fusion:** Concatenate image & text embeddings → 1024-D vector
* **Classifier block:** 2 × (LayerNorm → Dropout 0.5 → Linear (1024 → 512))
* **Heads:**
  * *Answer* (softmax over *N* frequent answers)
  * *Answer Type* (4-way)
  * *Answerability* (sigmoid)

Total trainable parameters: **≈ 2.7 M**.

---

## Pre-Processing Pipeline

1. **Question embeddings** – `clip.tokenize()` → `clip.encode_text()`
2. **Image embeddings** – CLIP preprocess transforms → `clip.encode_image()`
3. **Dataset split** – stratified by *answerability* & *answer_type* (train 90 %, val 5 %, test 5 %).
4. **Answer vocabulary** – top-K answers by frequency → label-index mapping.

Exception handling skips unreadable images & empty questions.

---

## Reproducing the Paper Figures

All charts are generated with `scripts/plot_curves.py` and saved into `docs/`.  Replace the checkpoint path to match your run, then embed the figures in your own reports with:
```markdown
![Training & Validation Loss](docs/loss_curve.png)
```

---

## Known Limitations & Future Work

* **Limited backbone** – ViT-B/32 leaves ~45 % of val questions unanswered correctly; upgrading to ViT-L/14 or RN50×64 is on the roadmap.
* **Unanswerable detection** – although answerability ≈ 80 %, false positives still exist.  Exploring contrastive image-text prompting may help.
* **Synonym merge** – simple majority-vote labels ignore synonyms (*“TV” vs “television”*).  We plan to adopt Soft VQA scoring.

Contributions & pull requests welcome!

---

## Citation

If you use this code, please cite our course project:

```bibtex
@misc{mandava2025vizwizclipvqa,
  title        = {Vision-Language Modeling for Real-World Accessibility: A CLIP-Based VQA System for VizWiz},
  author       = {Suman Mandava and Venkata Ramana Murthy Polisetty and Sharon Dev Saseendran},
  year         = {2025},
  howpublished = {Course Project, Texas A\&M University}
}
```

---

## License

This repository is licensed under the **MIT License**.  See [`LICENSE`](LICENSE) for details.

---

## Contributors

| Name | Role |
|------|------|
| **Suman Mandava** | Data pre-processing, training experiments |
| **Venkata Ramana Murthy Polisetty** | Model architecture & ablations |
| **Sharon Dev Saseendran** | Evaluation, visualization, README & report |

Special thanks to the VizWiz team for the dataset.
