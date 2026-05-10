# CTM3L: Consensus-Driven Tensor Learning for Multi-Source Multi-Instance Multi-Label Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 📌 Overview

**CTM3L** is a novel consensus-driven tensor learning framework designed for **Multi-source Multi-instance Multi-label Learning (MSMIML)**. It addresses the core challenges of propagating coarse bag-level annotations into reliable instance-level labels across heterogeneous data sources by leveraging **low-rank tensor representations** and **high-order instance–label–source interactions**.

Unlike existing methods that rely on pairwise correlation modeling and decouple fusion from classification, CTM3L jointly performs feature fusion and label prediction within a unified tensor space, achieving state-of-the-art performance on both bag- and instance-level metrics.

> **Key Innovation**: A tensor tri-mode projection (TMP) module that simultaneously:
> - Encodes heterogeneous source features into a multi-source feature tensor,
> - Projects into a latent semantic space for instance-level label prediction,
> - Estimates source contributions for adaptive consensus label fusion,
> - Enforces cross-source consistency via a self-consistent optimization loop.

---

## 💡 Usage

We provide a small synthetic multi-source multi-label dataset in `datasets/` for quick testing.
> For real-world experiments, please download full datasets from their official sources (see below).

> After downloading, place your dataset in `datasets/`.

> Run `main.py`, which contains the full implementation of the CTM3L model, its core algorithm, and evaluation code.

---


## 🔗 Datasets Used in the Paper

> **Please download the datasets from their official sources listed below.**

All datasets used in our experiments are publicly available. Please download them directly from the original providers:

| Dataset Name       | Description                                      | Official Download Link                                                                 |
|--------------------|--------------------------------------------------|----------------------------------------------------------------------------------------|
| **NOIZEUS**        | Speech corpus with noise-augmented utterances    | https://ecs.utdallas.edu/loizou/speech/noizeus/                                        |
| **3Sources**       | Multi-view news articles (BBC, Reuters, Guardian)| http://mlg.ucd.ie/datasets/3sources.html                                               |
| **human etc.**     | Mulan datasets                                   | http://mulan.sourceforge.net/datasets-mlc.html                                         |
| **Rugby**          | Event labeling in rugby match transcripts        | https://github.com/transientlunatic/rugby-data                                         |
| **ESC-50**         | Environmental sound classification (50 classes)  | https://github.com/karolpiczak/ESC-50                                                  |
| **mirflickr**      | Image dataset                                    | https://www.kaggle.com/datasets/matteocastrignano/mirflickr                            |
| **espgame**        | Image dataset                                    | https://www.kaggle.com/datasets/parhamsalar/espgame                                    |
| **Pascal VOC**     | Image classification & object detection          | https://pjreddie.com/projects/pascal-voc-dataset-mirror/                               |
| **Azotobacter_vinelandii etc.**      | Real-world organisms           | https://www.lamda.nju.edu.cn/data_MIMLprotein.ashx                                     |

> 💡 After downloading, organize the data as needed and specify the path via command-line arguments or config files (see usage below).

---

## 🚀 Getting Started

### Prerequisites
- Python ≥ 3.8
- PyTorch ≥ 1.10 (or NumPy for CPU-only version)
- Scikit-learn
- Matplotlib (optional, for visualization)

### Installation
```bash
git clone https://github.com/EalynChan/CTM3L.git
